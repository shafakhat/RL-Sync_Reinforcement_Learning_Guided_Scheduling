import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import java.util.concurrent.locks.*;
import java.io.*;
import java.nio.file.*;

public class RLSyncBenchmark {

    // ── Configuration (tuned for reproducibility in constrained env) ─────────
    static final int    CS_WORK_US            = 150;
    static final int    THINK_US_BASE         = 120;
    static final int    THINK_US_JITTER       = 80;
    static final long   NO_DELAY_THRESHOLD_NS = 500_000L; // 500µs = 1 scheduler quantum
    static final int    TRAIN_EPISODES        = 220;
    static final int    BURST_PER_EPISODE     = 3;
    static final int    EVAL_REQUESTS         = 100;
    static final int    SEEDS                 = 6;
    static final int[]  THREAD_COUNTS         = {2, 4, 6, 8};

    // ── Q-Learning ───────────────────────────────────────────────────────────
    static final double ALPHA        = 0.12;
    static final double GAMMA        = 0.92;
    static final double EPS_START    = 0.40;
    static final double EPS_END      = 0.04;
    static final int    MAX_WAIT     = 6;
    static final int    LOAD_BUCKETS = 4;
    static final int    NUM_STATES   = (MAX_WAIT + 1) * 2 * LOAD_BUCKETS;
    static final int    NUM_ACTIONS  = 2;  // 0=back-off, 1=admit

    static final int BACKOFF_BASE_US = 40;
    static final int BACKOFF_MAX_US  = 600;

    // ── Busy-wait sleep for sub-ms precision ────────────────────────────────
    static void sleepMicros(long us) throws InterruptedException {
        if (us <= 0) return;
        long deadline = System.nanoTime() + us * 1000L;
        long remainNs = deadline - System.nanoTime();
        if (remainNs > 1_500_000L) Thread.sleep(remainNs / 1_500_000L);
        while (System.nanoTime() < deadline) {
            if (Thread.interrupted()) throw new InterruptedException();
            Thread.onSpinWait();
        }
    }

    // ── Metrics ──────────────────────────────────────────────────────────────
    static class Metrics {
        final String name;
        final List<Long> samples = Collections.synchronizedList(new ArrayList<>());
        final AtomicInteger contended = new AtomicInteger(0);
        final AtomicInteger total     = new AtomicInteger(0);
        Metrics(String n) { name = n; }

        void record(long waitNs, boolean c) {
            samples.add(waitNs); total.incrementAndGet();
            if (c) contended.incrementAndGet();
        }
        double meanMs()  { return samples.stream().mapToLong(v->v).average().orElse(0)/1e6; }
        double stdMs()   {
            double m = meanMs()*1e6;
            double v = samples.stream().mapToDouble(w->(w-m)*(w-m)).average().orElse(0);
            return Math.sqrt(v)/1e6;
        }
        double p99Ms()   {
            long[] s = samples.stream().mapToLong(v->v).sorted().toArray();
            return s.length==0?0:s[Math.min((int)(s.length*0.99),s.length-1)]/1e6;
        }
        double zdr()     { return 100.0*samples.stream().filter(w->w<NO_DELAY_THRESHOLD_NS).count()/Math.max(1,samples.size()); }
        double contRate(){ return 100.0*contended.get()/Math.max(1,total.get()); }
    }

    // ── CSLock interface ─────────────────────────────────────────────────────
    interface CSLock { long acquire(boolean rec) throws InterruptedException; void release(); Metrics metrics(); }

    // ── Q-Table (extended state: nWait × csOcc × loadBucket) ─────────────────
    static class QTable {
        final double[][] q = new double[NUM_STATES][NUM_ACTIONS];
        volatile double emaWaitNs = 0;
        static final double EMA_A = 0.08;
        synchronized void updateEMA(long w) { emaWaitNs = EMA_A*w+(1-EMA_A)*emaWaitNs; }
        int lb() {
            long w = (long)emaWaitNs;
            if (w<200_000L) return 0; if (w<1_000_000L) return 1;
            if (w<5_000_000L) return 2; return 3;
        }
        int sid(int nw, boolean occ) { return Math.min(nw,MAX_WAIT)*2*LOAD_BUCKETS+(occ?1:0)*LOAD_BUCKETS+lb(); }
        int best(int s) { return q[s][1]>q[s][0]?1:0; }
        void update(int s, int a, double r, int sp) {
            double mq=Math.max(q[sp][0],q[sp][1]); q[s][a]+=ALPHA*(r+GAMMA*mq-q[s][a]);
        }
        void exportCSV(String p) throws IOException {
            try(PrintWriter pw=new PrintWriter(p)) {
                pw.println("nWaiting,csOccupied,loadBucket,Q_backoff,Q_admit,policy");
                for(int nw=0;nw<=MAX_WAIT;nw++) for(int cs=0;cs<=1;cs++) for(int lb=0;lb<LOAD_BUCKETS;lb++){
                    int s=nw*2*LOAD_BUCKETS+cs*LOAD_BUCKETS+lb;
                    pw.printf("%d,%d,%d,%.4f,%.4f,%s%n",nw,cs,lb,q[s][0],q[s][1],q[s][1]>q[s][0]?"ADMIT":"BACKOFF");
                }
            }
        }
    }

    // ── RL Arbiter ────────────────────────────────────────────────────────────
    static class RLArbiter implements CSLock {
        final Semaphore sem = new Semaphore(1,true);
        final AtomicInteger nWait = new AtomicInteger(0);
        final AtomicBoolean occ  = new AtomicBoolean(false);
        final QTable qt; final double eps; final boolean train;
        final Metrics met; final List<Double> convLog;
        long epWaitSum=0; int epCount=0; final Object epLock=new Object();

        RLArbiter(QTable qt,double eps,boolean train,String tag,List<Double> cl){
            this.qt=qt;this.eps=eps;this.train=train;met=new Metrics(tag);convLog=cl;
        }
        public long acquire(boolean rec) throws InterruptedException {
            int nw=nWait.incrementAndGet(); boolean csOcc=occ.get();
            boolean contended=csOcc||(nw>1);
            int s=qt.sid(nw,csOcc);
            ThreadLocalRandom tlr=ThreadLocalRandom.current();
            int a=(train&&tlr.nextDouble()<eps)?tlr.nextInt(2):qt.best(s);
            long t0=System.nanoTime();
            if(a==0) sleepMicros(Math.min(BACKOFF_BASE_US*(1+nw),BACKOFF_MAX_US));
            sem.acquire();
            long wNs=System.nanoTime()-t0;
            nWait.decrementAndGet(); occ.set(true); qt.updateEMA(wNs);
            if(train){ int sp=qt.sid(nWait.get(),true); qt.update(s,a,-wNs/1e6,sp);
                synchronized(epLock){epWaitSum+=wNs;epCount++;} }
            if(rec) met.record(wNs,contended);
            return wNs;
        }
        public void release(){ occ.set(false); sem.release(); }
        public Metrics metrics(){ return met; }
        void flushEp(){ synchronized(epLock){ if(epCount>0&&convLog!=null){
            convLog.add(epWaitSum/1e6/epCount); epWaitSum=0;epCount=0; } } }
    }

    // ── Fair ReentrantLock ───────────────────────────────────────────────────
    static class FairMutex implements CSLock {
        final ReentrantLock lk=new ReentrantLock(true); final Metrics met;
        FairMutex(String t){met=new Metrics(t);}
        public long acquire(boolean rec) throws InterruptedException {
            boolean c=lk.hasQueuedThreads(); long t0=System.nanoTime();
            lk.lockInterruptibly(); long w=System.nanoTime()-t0;
            if(rec) met.record(w,c); return w;
        }
        public void release(){ lk.unlock(); }
        public Metrics metrics(){ return met; }
    }

    // ── CLH Queue Lock ────────────────────────────────────────────────────────
    static class CLHLock implements CSLock {
        static class Node { volatile boolean locked=false; }
        final AtomicReference<Node> tail=new AtomicReference<>(new Node());
        final ThreadLocal<Node> myNode=ThreadLocal.withInitial(Node::new);
        final ThreadLocal<Node> myPred=new ThreadLocal<>();
        final Metrics met;
        CLHLock(String t){met=new Metrics(t);}
        public long acquire(boolean rec) throws InterruptedException {
            Node nd=myNode.get(); nd.locked=true;
            Node pred=tail.getAndSet(nd); myPred.set(pred);
            long t0=System.nanoTime(); boolean c=pred.locked;
            while(pred.locked){ if(Thread.interrupted()){nd.locked=false;throw new InterruptedException();} Thread.onSpinWait(); }
            long w=System.nanoTime()-t0; if(rec) met.record(w,c); return w;
        }
        public void release(){ Node nd=myNode.get(); nd.locked=false; myNode.set(myPred.get()); }
        public Metrics metrics(){ return met; }
    }

    // ── Exponential Back-off Lock ─────────────────────────────────────────────
    static class ExpBackoff implements CSLock {
        final AtomicBoolean st=new AtomicBoolean(false); final Metrics met;
        ExpBackoff(String t){met=new Metrics(t);}
        public long acquire(boolean rec) throws InterruptedException {
            ThreadLocalRandom tlr=ThreadLocalRandom.current();
            long t0=System.nanoTime(); boolean c=st.get(); long d=BACKOFF_BASE_US;
            while(!st.compareAndSet(false,true)){
                if(Thread.interrupted()) throw new InterruptedException();
                sleepMicros(tlr.nextLong(d)); d=Math.min(d*2,BACKOFF_MAX_US);
            }
            long w=System.nanoTime()-t0; if(rec) met.record(w,c); return w;
        }
        public void release(){ st.set(false); }
        public Metrics metrics(){ return met; }
    }

    // ── Worker ───────────────────────────────────────────────────────────────
    static class Worker implements Runnable {
        final CSLock lk; final int reqs; final CountDownLatch done; final boolean rec;
        final AtomicInteger inside; final AtomicBoolean violated;
        Worker(CSLock l,int r,CountDownLatch d,boolean rc,AtomicInteger in,AtomicBoolean v){
            lk=l;reqs=r;done=d;rec=rc;inside=in;violated=v;
        }
        public void run(){
            ThreadLocalRandom tlr=ThreadLocalRandom.current();
            try { for(int i=0;i<reqs;i++){
                sleepMicros(THINK_US_BASE+tlr.nextLong(THINK_US_JITTER));
                lk.acquire(rec);
                int n=inside.incrementAndGet(); if(n>1) violated.set(true);
                sleepMicros(CS_WORK_US); inside.decrementAndGet(); lk.release();
            }} catch(InterruptedException e){Thread.currentThread().interrupt();}
            finally{done.countDown();}
        }
    }

    static boolean runWorkers(CSLock lk, int nT, int reqs, boolean rec) throws Exception {
        AtomicInteger inside=new AtomicInteger(0); AtomicBoolean viol=new AtomicBoolean(false);
        CountDownLatch latch=new CountDownLatch(nT);
        ExecutorService ex=Executors.newFixedThreadPool(nT);
        for(int t=0;t<nT;t++) ex.submit(new Worker(lk,reqs,latch,rec,inside,viol));
        latch.await(); ex.shutdownNow(); return !viol.get();
    }

    static QTable trainRL(int nT, long seed, List<Double> conv) throws Exception {
        QTable qt=new QTable();
        for(int ep=0;ep<TRAIN_EPISODES;ep++){
            double p=(double)ep/TRAIN_EPISODES;
            double eps=EPS_START-p*(EPS_START-EPS_END);
            RLArbiter arb=new RLArbiter(qt,eps,true,"tr",conv);
            runWorkers(arb,nT,BURST_PER_EPISODE,false);
            arb.flushEp();
        }
        return qt;
    }

    // ── Stats ────────────────────────────────────────────────────────────────
    static double mean(double[]a){double s=0;for(double v:a)s+=v;return s/a.length;}
    static double std(double[]a){double m=mean(a),s=0;for(double v:a)s+=(v-m)*(v-m);return Math.sqrt(s/a.length);}
    static double ci95(double[]a){return 2.447*std(a)/Math.sqrt(a.length);} // t_{6,0.025}=2.447
    static double welchT(double[]a,double[]b){
        double ma=mean(a),mb=mean(b),va=0,vb=0;
        for(double v:a)va+=(v-ma)*(v-ma); for(double v:b)vb+=(v-mb)*(v-mb);
        va/=(a.length-1);vb/=(b.length-1);
        return (ma-mb)/Math.sqrt(va/a.length+vb/b.length);
    }
    static double cohensD(double[]a,double[]b){
        double p=Math.sqrt((std(a)*std(a)+std(b)*std(b))/2.0);
        return p<1e-12?0:(mean(a)-mean(b))/p;
    }

    // ── CSV writers ──────────────────────────────────────────────────────────
    // sweep_results.csv
    static void writeSweep(String path, Map<String,Map<Integer,double[][]>> R) throws IOException {
        try(PrintWriter pw=new PrintWriter(path)){
            pw.println("threads,method,meanWait_ms,ci95_ms,stdWait_ms,p99_ms,zdr_pct,contention_pct");
            for(var me:R.entrySet()) for(var te:me.getValue().entrySet()){
                int tc=te.getKey(); double[][]d=te.getValue();
                double[]mw=new double[d.length],sw=new double[d.length],
                        p=new double[d.length],z=new double[d.length],c=new double[d.length];
                for(int i=0;i<d.length;i++){mw[i]=d[i][0];sw[i]=d[i][1];p[i]=d[i][2];z[i]=d[i][3];c[i]=d[i][4];}
                pw.printf("%d,%s,%.5f,%.5f,%.5f,%.5f,%.3f,%.3f%n",
                    tc,me.getKey(),mean(mw),ci95(mw),mean(sw),mean(p),mean(z),mean(c));
            }
        }
    }
    // convergence.csv
    static void writeConv(String path,List<Double>rl,double mxBaseline) throws IOException {
        try(PrintWriter pw=new PrintWriter(path)){
            pw.println("episode,rl_mean_wait_ms,mutex_baseline_ms");
            for(int i=0;i<rl.size();i++) pw.printf("%d,%.5f,%.5f%n",i+1,rl.get(i),mxBaseline);
        }
    }
    // stats comparison CSV
    static void writeStats(String path, Map<String,Map<Integer,double[][]>>R,
                           String mA, String mB) throws IOException {
        try(PrintWriter pw=new PrintWriter(path)){
            pw.println("threads,metric,mean_A,mean_B,delta_pct,welch_t,cohens_d");
            String[]mn={"meanWait","stdWait","p99","zdr","contention"};
            for(int tc:THREAD_COUNTS){
                double[][]ra=R.get(mA).get(tc),rb=R.get(mB).get(tc);
                if(ra==null||rb==null)continue;
                for(int mi=0;mi<mn.length;mi++){
                    final int mif=mi;
                    double[]a=Arrays.stream(ra).mapToDouble(r->r[mif]).toArray();
                    double[]b=Arrays.stream(rb).mapToDouble(r->r[mif]).toArray();
                    double imp=mean(b)!=0?100.0*(mean(b)-mean(a))/Math.abs(mean(b)):0;
                    pw.printf("%d,%s,%.5f,%.5f,%.2f,%.4f,%.4f%n",
                        tc,mn[mi],mean(a),mean(b),imp,welchT(a,b),cohensD(a,b));
                }
            }
        }
    }

    // ── Main ─────────────────────────────────────────────────────────────────
    public static void main(String[]args) throws Exception {
        String dataDir=args.length>0?args[0]:"./mlsync_experiment_data/data";
        Files.createDirectories(Paths.get(dataDir));

        System.out.println("═══════════════════════════════════════════════════════════");
        System.out.println("  RL-Sync : RL Synchronisation       				");
        System.out.println("═══════════════════════════════════════════════════════════");
        System.out.printf("  CS=%dµs  Think=%d±%dµs  ZDR-thresh=%dµs  Seeds=%d      %n",
            CS_WORK_US,THINK_US_BASE,THINK_US_JITTER,(int)(NO_DELAY_THRESHOLD_NS/1000),SEEDS);
        System.out.printf("  Threads: %s  TrainEp: %d               		%n",
            Arrays.toString(THREAD_COUNTS),TRAIN_EPISODES);
        System.out.println("═══════════════════════════════════════════════════════════");

        String[]methods={"RL-Sync","FairMutex","CLH","ExpBackoff"};
        Map<String,Map<Integer,double[][]>>R=new LinkedHashMap<>();
        for(String m:methods)R.put(m,new TreeMap<>());

        List<Double> rlConv=new ArrayList<>();
        double mxBaseline=Double.NaN;
        QTable savedQT=null;

        for(int nT:THREAD_COUNTS){
            System.out.printf("%n── Threads = %2d ─────────────────────────────────────────%n",nT);
            double[][]rls=new double[SEEDS][],mxs=new double[SEEDS][],
                     clhs=new double[SEEDS][],ebs=new double[SEEDS][];

            for(int s=0;s<SEEDS;s++){
                long seed=(s+1)*7919L+nT*31L;
                System.out.printf("  Seed %2d/%d  Train...",s+1,SEEDS);System.out.flush();

                boolean captureConv=(nT==THREAD_COUNTS[2]&&s==0);
                List<Double>conv=captureConv?rlConv:null;
                QTable qt=trainRL(nT,seed,conv);
                if(captureConv)savedQT=qt;

                System.out.print("RL...");System.out.flush();
                RLArbiter rlEv=new RLArbiter(qt,0.0,false,"rl"+nT,null);
                boolean meOk=runWorkers(rlEv,nT,EVAL_REQUESTS,true);
                Metrics rm=rlEv.metrics();
                rls[s]=new double[]{rm.meanMs(),rm.stdMs(),rm.p99Ms(),rm.zdr(),rm.contRate(),meOk?0:1};

                System.out.print("Mutex...");System.out.flush();
                FairMutex mx=new FairMutex("mx"+nT);
                runWorkers(mx,nT,EVAL_REQUESTS,true);
                Metrics mm=mx.metrics();
                mxs[s]=new double[]{mm.meanMs(),mm.stdMs(),mm.p99Ms(),mm.zdr(),mm.contRate(),0};
                if(captureConv&&s==0) mxBaseline=mm.meanMs();

                System.out.print("CLH...");System.out.flush();
                CLHLock clh=new CLHLock("clh"+nT);
                runWorkers(clh,nT,EVAL_REQUESTS,true);
                Metrics cm=clh.metrics();
                clhs[s]=new double[]{cm.meanMs(),cm.stdMs(),cm.p99Ms(),cm.zdr(),cm.contRate(),0};

                System.out.print("ExpBk...");System.out.flush();
                ExpBackoff eb=new ExpBackoff("eb"+nT);
                runWorkers(eb,nT,EVAL_REQUESTS,true);
                Metrics em=eb.metrics();
                ebs[s]=new double[]{em.meanMs(),em.stdMs(),em.p99Ms(),em.zdr(),em.contRate(),0};

                System.out.printf(" RL[%.2fms,ZDR=%.0f%%,ME=%s] Mx[%.2fms,ZDR=%.0f%%]%n",
                    rls[s][0],rls[s][3],meOk?"OK":"FAIL",mxs[s][0],mxs[s][3]);

                if(nT==THREAD_COUNTS[2]&&s==SEEDS-1&&savedQT!=null)
                    savedQT.exportCSV(dataDir+"/qtable.csv");
            }
            R.get("RL-Sync").put(nT,rls); R.get("FairMutex").put(nT,mxs);
            R.get("CLH").put(nT,clhs);   R.get("ExpBackoff").put(nT,ebs);
        }

        // ME check
        System.out.println("\n── Mutual Exclusion Verification ──────────────────────────");
        boolean allOk=true;
        for(int nT:THREAD_COUNTS){
            boolean ok=Arrays.stream(R.get("RL-Sync").get(nT)).allMatch(r->r[5]==0);
            System.out.printf("  Threads=%2d : %s%n",nT,ok?"PASS ✓":"FAIL ✗");
            allOk&=ok;
        }

        // Write CSVs
        writeSweep(dataDir+"/sweep_results.csv",R);
        if(!rlConv.isEmpty())writeConv(dataDir+"/convergence.csv",rlConv,
            Double.isNaN(mxBaseline)?2.0:mxBaseline);
        writeStats(dataDir+"/stats_rl_vs_mutex.csv",R,"RL-Sync","FairMutex");
        writeStats(dataDir+"/stats_rl_vs_clh.csv",R,"RL-Sync","CLH");
        writeStats(dataDir+"/stats_rl_vs_eb.csv",R,"RL-Sync","ExpBackoff");

        // Summary at 6 threads
        int tc6=6;
        System.out.println("\n════ SUMMARY (threads=6, mean±CI95 over "+SEEDS+" seeds) ══════════");
        System.out.printf(" %-11s %9s %9s %9s %9s %n","Method","Wait(ms)","P99(ms)","ZDR(%)","Cont(%)");
        System.out.println("═══════════════════════════════════════════════════════════");
        for(String m:methods){
            double[][]d=R.get(m).get(tc6);
            double[]mw=Arrays.stream(d).mapToDouble(r->r[0]).toArray();
            double[]p9=Arrays.stream(d).mapToDouble(r->r[2]).toArray();
            double[]zd=Arrays.stream(d).mapToDouble(r->r[3]).toArray();
            double[]ct=Arrays.stream(d).mapToDouble(r->r[4]).toArray();
            System.out.printf(" %-11s %5.2f±%.2f %9.2f %9.1f %9.1f %n",
                m,mean(mw),ci95(mw),mean(p9),mean(zd),mean(ct));
        }

        // Statistical significance vs FairMutex at 6 threads
        System.out.println("═══════════════════════════════════════════════════════════");
        System.out.println(" RL-Sync vs FairMutex (6 threads): Welch t / Cohen d       ");
        double[][]rl6=R.get("RL-Sync").get(tc6),mx6=R.get("FairMutex").get(tc6);
        double[]rlW=Arrays.stream(rl6).mapToDouble(r->r[0]).toArray();
        double[]mxW=Arrays.stream(mx6).mapToDouble(r->r[0]).toArray();
        double[]rlZ=Arrays.stream(rl6).mapToDouble(r->r[3]).toArray();
        double[]mxZ=Arrays.stream(mx6).mapToDouble(r->r[3]).toArray();
        System.out.printf("  Wait: t=%.3f, d=%.3f, delta=%.1f%%                        %n",
            welchT(rlW,mxW),cohensD(rlW,mxW),mean(mxW)!=0?100*(mean(mxW)-mean(rlW))/mean(mxW):0);
        System.out.printf("  ZDR:  t=%.3f, d=%.3f, delta=%.1f pp                       %n",
            welchT(rlZ,mxZ),cohensD(rlZ,mxZ),mean(rlZ)-mean(mxZ));
        System.out.println("═══════════════════════════════════════════════════════════");
        System.out.println("\nData → "+dataDir);
    }
}
