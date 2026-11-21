/*
 * pattern_lu.c -- sparse LU of F complex matrices that share one sparsity pattern.
 *
 * Used by differometor/fast.py (switch `fast.NATIVE_LU`) for the signal systems of a frequency
 * sweep. All F matrices are factorized with ONE pivot sequence. Every arithmetic step is then a
 * loop over the values of one matrix entry at consecutive frequencies (stored contiguously),
 * which the CPU runs as a plain vector loop: no index work, no pivot search and no symbolic
 * analysis per frequency.
 *
 * Two phases:
 *
 * 1. Analysis on a sample of the frequencies (`analyse`). A right-looking elimination chooses
 *    the pivots on the fly: the entry with the lowest Markowitz count among those that pass a
 *    stability test at every sampled frequency,
 *
 *        |a_rc| >= u * max_j |a_rj|   and   |a_rc| >= u * max_i |a_ic|        (u = u_select)
 *
 *    (threshold rook pivoting; rows and columns are those of the active submatrix). Row or
 *    column singletons need no elimination arithmetic and are accepted when they are nonzero.
 *    The result is the pivot sequence, the pattern of L and U and the list of update operations.
 *
 * 2. Numeric factorization of all frequencies with that sequence (`factor_chunk`). Every pivot
 *    is checked against the same two conditions at EVERY frequency (u = u_verify <= u_select).
 *    Multipliers are therefore bounded by 1/u_verify in both factors at every frequency, the
 *    usual threshold guarantee of sparse direct solvers, here with respect to the scale of the
 *    row and of the column.
 *
 * If no entry is admissible in phase 1, or a pivot fails the check in phase 2, the function
 * reports failure and the caller uses its general solver (SuperLU, partial pivoting per
 * frequency). The values are overwritten in place, so the caller has to restore them first.
 *
 * Arithmetic: IEEE double, no fused multiply-add, the same operation order in the scalar and in
 * the AVX2 code, and the result does not depend on the chunk size. Results are therefore
 * bit-identical on every x86-64 CPU. AVX2 is selected at run time.
 *
 * Build: see build.sh (gcc -O2, baseline x86-64, no -march=native). No dependencies except libc.
 */
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <limits.h>
#include <immintrin.h>
#include <time.h>

#define PLU_OK 0
#define PLU_NO_PIVOT 1  /* analysis: no admissible pivot (singular, or no shared pivot is stable) */
#define PLU_NO_ROOM 2   /* fill-in exceeds the capacity of the value store, or bad arguments */
#define PLU_NO_MEMORY 3
#define PLU_UNSTABLE 4  /* numeric phase: a pivot fails the stability check at some frequency */

/* ------------------------------------------------------------------------------------------
 * Kernels over n consecutive values of an entry. Complex numbers are interleaved (re, im).
 * product(a, b) = (ar*br - ai*bi, ar*bi + ai*br), each product and sum rounded separately.
 * ---------------------------------------------------------------------------------------- */

typedef struct {
    void (*mulsub)(double *y, const double *a, const double *b, int n);  /* y -= a*b */
    void (*negmul)(double *y, const double *a, const double *b, int n);  /* y = 0 - a*b */
    void (*mul)(double *y, const double *a, const double *b, int n);     /* y = a*b */
    int (*recip)(double *y, int n);                                      /* y = 1/y; nonzero if out of range */
    void (*absmax)(double *mx, const double *a, int n);                  /* mx = max(|a|^2, mx), stored twice per value */
    int (*level)(const double *mx, const double *a, int n, double u2);   /* 2: |a|^2 >= mx everywhere, 1: >= u2*mx, else 0 */
} kernels_t;

/* 1/z is computed as conj(z)/|z|^2, which needs |z|^2 well inside the double range. */
#define RANGE_LOW 1e-280
#define RANGE_HIGH 1e280

static void mulsub_c(double *y, const double *a, const double *b, int n) {
    for (int f = 0; f < 2 * n; f += 2) {
        double ar = a[f], ai = a[f + 1], br = b[f], bi = b[f + 1];
        double p1 = ar * br, p2 = ai * bi, p3 = ar * bi, p4 = ai * br;
        y[f] = y[f] - (p1 - p2);
        y[f + 1] = y[f + 1] - (p3 + p4);
    }
}
static void negmul_c(double *y, const double *a, const double *b, int n) {
    for (int f = 0; f < 2 * n; f += 2) {
        double ar = a[f], ai = a[f + 1], br = b[f], bi = b[f + 1];
        double p1 = ar * br, p2 = ai * bi, p3 = ar * bi, p4 = ai * br;
        y[f] = 0.0 - (p1 - p2);
        y[f + 1] = 0.0 - (p3 + p4);
    }
}
static void mul_c(double *y, const double *a, const double *b, int n) {
    for (int f = 0; f < 2 * n; f += 2) {
        double ar = a[f], ai = a[f + 1], br = b[f], bi = b[f + 1];
        double p1 = ar * br, p2 = ai * bi, p3 = ar * bi, p4 = ai * br;
        y[f] = p1 - p2;
        y[f + 1] = p3 + p4;
    }
}
static int recip_c(double *y, int n) {
    int ok = 1;
    for (int f = 0; f < 2 * n; f += 2) {
        double s1 = y[f] * y[f], s2 = y[f + 1] * y[f + 1];
        double d = s1 + s2;
        ok &= (d >= RANGE_LOW) & (d <= RANGE_HIGH);
        y[f] = y[f] / d;
        y[f + 1] = -(y[f + 1] / d);
    }
    return !ok;
}
static void absmax_c(double *mx, const double *a, int n) {
    for (int f = 0; f < 2 * n; f += 2) {
        double s1 = a[f] * a[f], s2 = a[f + 1] * a[f + 1];
        double d = s1 + s2;
        d = d > mx[f] ? d : mx[f];
        mx[f] = d;
        mx[f + 1] = d;
    }
}
static int level_c(const double *mx, const double *a, int n, double u2) {
    int all1 = 1, all2 = 1;
    for (int f = 0; f < 2 * n; f += 2) {
        double s1 = a[f] * a[f], s2 = a[f + 1] * a[f + 1];
        double d = s1 + s2;
        int usable = (d > 0.0) & (d <= DBL_MAX);
        all2 &= usable & (d >= mx[f]);
        all1 &= usable & (d >= u2 * mx[f]);
    }
    return all2 ? 2 : (all1 ? 1 : 0);
}

#define TARGET_AVX2 __attribute__((target("avx2")))

TARGET_AVX2 static inline __m256d product_avx2(__m256d a, __m256d b) {
    __m256d are = _mm256_movedup_pd(a);       /* ar, ar */
    __m256d aim = _mm256_permute_pd(a, 0xF);  /* ai, ai */
    __m256d bsw = _mm256_permute_pd(b, 0x5);  /* bi, br */
    return _mm256_addsub_pd(_mm256_mul_pd(are, b), _mm256_mul_pd(aim, bsw));
}
TARGET_AVX2 static void mulsub_avx2(double *y, const double *a, const double *b, int n) {
    int f = 0, end = 2 * n;
    for (; f + 4 <= end; f += 4) {
        __m256d p = product_avx2(_mm256_loadu_pd(a + f), _mm256_loadu_pd(b + f));
        _mm256_storeu_pd(y + f, _mm256_sub_pd(_mm256_loadu_pd(y + f), p));
    }
    if (f < end) mulsub_c(y + f, a + f, b + f, (end - f) / 2);
}
TARGET_AVX2 static void negmul_avx2(double *y, const double *a, const double *b, int n) {
    int f = 0, end = 2 * n;
    for (; f + 4 <= end; f += 4) {
        __m256d p = product_avx2(_mm256_loadu_pd(a + f), _mm256_loadu_pd(b + f));
        _mm256_storeu_pd(y + f, _mm256_sub_pd(_mm256_setzero_pd(), p));
    }
    if (f < end) negmul_c(y + f, a + f, b + f, (end - f) / 2);
}
TARGET_AVX2 static void mul_avx2(double *y, const double *a, const double *b, int n) {
    int f = 0, end = 2 * n;
    for (; f + 4 <= end; f += 4)
        _mm256_storeu_pd(y + f, product_avx2(_mm256_loadu_pd(a + f), _mm256_loadu_pd(b + f)));
    if (f < end) mul_c(y + f, a + f, b + f, (end - f) / 2);
}
TARGET_AVX2 static int recip_avx2(double *y, int n) {
    int f = 0, end = 2 * n;
    const __m256d low = _mm256_set1_pd(RANGE_LOW), high = _mm256_set1_pd(RANGE_HIGH);
    const __m256d flip = _mm256_set_pd(-0.0, 0.0, -0.0, 0.0);
    __m256d ok = _mm256_castsi256_pd(_mm256_set1_epi64x(-1));
    for (; f + 4 <= end; f += 4) {
        __m256d v = _mm256_loadu_pd(y + f);
        __m256d sq = _mm256_mul_pd(v, v);
        __m256d d = _mm256_hadd_pd(sq, sq);
        ok = _mm256_and_pd(ok, _mm256_and_pd(_mm256_cmp_pd(d, low, _CMP_GE_OQ), _mm256_cmp_pd(d, high, _CMP_LE_OQ)));
        _mm256_storeu_pd(y + f, _mm256_xor_pd(_mm256_div_pd(v, d), flip));
    }
    int bad = _mm256_movemask_pd(ok) != 0xF;
    if (f < end) bad |= recip_c(y + f, (end - f) / 2);
    return bad;
}
TARGET_AVX2 static void absmax_avx2(double *mx, const double *a, int n) {
    int f = 0, end = 2 * n;
    for (; f + 4 <= end; f += 4) {
        __m256d v = _mm256_loadu_pd(a + f);
        __m256d sq = _mm256_mul_pd(v, v);
        __m256d d = _mm256_hadd_pd(sq, sq);
        _mm256_storeu_pd(mx + f, _mm256_max_pd(d, _mm256_loadu_pd(mx + f)));  /* d > mx ? d : mx */
    }
    if (f < end) absmax_c(mx + f, a + f, (end - f) / 2);
}
TARGET_AVX2 static int level_avx2(const double *mx, const double *a, int n, double u2) {
    int f = 0, end = 2 * n;
    const __m256d zero = _mm256_setzero_pd(), top = _mm256_set1_pd(DBL_MAX), vu2 = _mm256_set1_pd(u2);
    __m256d all1 = _mm256_castsi256_pd(_mm256_set1_epi64x(-1)), all2 = all1;
    for (; f + 4 <= end; f += 4) {
        __m256d v = _mm256_loadu_pd(a + f), m = _mm256_loadu_pd(mx + f);
        __m256d sq = _mm256_mul_pd(v, v);
        __m256d d = _mm256_hadd_pd(sq, sq);
        __m256d usable = _mm256_and_pd(_mm256_cmp_pd(d, zero, _CMP_GT_OQ), _mm256_cmp_pd(d, top, _CMP_LE_OQ));
        all2 = _mm256_and_pd(all2, _mm256_and_pd(usable, _mm256_cmp_pd(d, m, _CMP_GE_OQ)));
        all1 = _mm256_and_pd(all1, _mm256_and_pd(usable, _mm256_cmp_pd(d, _mm256_mul_pd(vu2, m), _CMP_GE_OQ)));
    }
    int r2 = _mm256_movemask_pd(all2) == 0xF, r1 = _mm256_movemask_pd(all1) == 0xF;
    if (f < end) {
        int tail = level_c(mx + f, a + f, (end - f) / 2, u2);
        r2 &= tail == 2;
        r1 &= tail >= 1;
    }
    return r2 ? 2 : (r1 ? 1 : 0);
}

static const kernels_t KERNELS_C = {mulsub_c, negmul_c, mul_c, recip_c, absmax_c, level_c};
static const kernels_t KERNELS_AVX2 = {mulsub_avx2, negmul_avx2, mul_avx2, recip_avx2, absmax_avx2, level_avx2};
static const kernels_t *K = &KERNELS_C;
static int simd_level = -1;  /* -1: not selected yet, 0: scalar, 2: AVX2 */

/* Select the kernels: `want` < 0 picks the best the CPU supports, 0 forces the scalar code. */
int plu_select(int want) {
    int have = 0;
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx2")) have = 2;
    if (want >= 0 && want < have) have = want;
    K = have >= 2 ? &KERNELS_AVX2 : &KERNELS_C;
    simd_level = have >= 2 ? 2 : 0;
    return simd_level;
}

/* ------------------------------------------------------------------------------------------
 * Data
 * ---------------------------------------------------------------------------------------- */

typedef struct plu {
    int m, F, cap, used, chunk;
    double *val;  /* (cap, F) complex, owned by the caller; holds L, U and the inverse pivots */
    int *prow, *pcol, *pid;                      /* per step: pivot row, pivot column, entry of the pivot */
    unsigned char *punit;                        /* pivot is exactly 1 at every frequency */
    int *lptr, *lrow, *lid, *uptr, *ucol, *uid;  /* column k of L and row k of U: index and entry */
    int *optr, *otarget;                         /* updates of step k, L entry by L entry; ~t marks fill-in */
    int nl, nu, nops, cap_ops;
    long evals, searched, ns_analyse, ns_numeric;
    double *work;          /* 2 * m * F complex, for solves in the (F, m, k) layout */
    unsigned char *flags;  /* 3 * m */
    void *block;
} plu_t;

/* State of the analysis: the active submatrix as linked row and column lists. */
typedef struct {
    plu_t *s;
    double *av;  /* (cap, n) values at the sampled frequencies */
    int n;
    int *erow, *ecol, *rnext, *rprev, *cnext, *cprev;         /* per entry */
    int *rhead, *rcnt, *chead, *ccnt;                         /* per row / column */
    int *rbhead, *rbnext, *rbprev, *cbhead, *cbnext, *cbprev; /* rows and columns bucketed by count */
    int *pos;
    unsigned char *rdirty, *cdirty, *rlev, *clev;
    double *mx;
    double u2;
} work_t;

#define VAL(s, e) ((s)->val + (size_t)(e) * 2 * (s)->F)
#define SAMPLE(w, e) ((w)->av + (size_t)(e) * 2 * (w)->n)

static inline void bucket_remove(int *head, int *next, int *prev, int item, int count) {
    if (prev[item] >= 0) next[prev[item]] = next[item]; else head[count] = next[item];
    if (next[item] >= 0) prev[next[item]] = prev[item];
}
static inline void bucket_insert(int *head, int *next, int *prev, int item, int count) {
    prev[item] = -1;
    next[item] = head[count];
    if (head[count] >= 0) prev[head[count]] = item;
    head[count] = item;
}

static int is_unit(const double *a, int n) {
    for (int f = 0; f < 2 * n; f += 2)
        if (a[f] != 1.0 || a[f + 1] != 0.0) return 0;
    return 1;
}
static int is_zero(const double *a, int n) {
    for (int f = 0; f < 2 * n; f++)
        if (a[f] != 0.0) return 0;
    return 1;
}

/* ------------------------------------------------------------------------------------------
 * Phase 1: pivot sequence and structure from the sampled frequencies
 * ---------------------------------------------------------------------------------------- */

static void eval_row(work_t *w, int r) {
    memset(w->mx, 0, sizeof(double) * 2 * w->n);
    for (int e = w->rhead[r]; e >= 0; e = w->rnext[e]) K->absmax(w->mx, SAMPLE(w, e), w->n);
    for (int e = w->rhead[r]; e >= 0; e = w->rnext[e]) w->rlev[e] = (unsigned char)K->level(w->mx, SAMPLE(w, e), w->n, w->u2);
    w->rdirty[r] = 0;
    w->s->evals++;
}
static void eval_col(work_t *w, int c) {
    memset(w->mx, 0, sizeof(double) * 2 * w->n);
    for (int e = w->chead[c]; e >= 0; e = w->cnext[e]) K->absmax(w->mx, SAMPLE(w, e), w->n);
    for (int e = w->chead[c]; e >= 0; e = w->cnext[e]) w->clev[e] = (unsigned char)K->level(w->mx, SAMPLE(w, e), w->n, w->u2);
    w->cdirty[c] = 0;
    w->s->evals++;
}

/* 0: not admissible, 1: passes the thresholds, 2: largest entry of its row and column everywhere. */
static inline int entry_level(work_t *w, int e) {
    int r = w->erow[e], c = w->ecol[e];
    if (w->rcnt[r] == 1) {  /* row singleton: nothing to eliminate with, only a nonzero pivot is needed */
        if (w->rdirty[r]) eval_row(w, r);
        return w->rlev[e];
    }
    if (w->ccnt[c] == 1) {
        if (w->cdirty[c]) eval_col(w, c);
        return w->clev[e];
    }
    if (w->rdirty[r]) eval_row(w, r);
    if (!w->rlev[e]) return 0;
    if (w->cdirty[c]) eval_col(w, c);
    return w->rlev[e] < w->clev[e] ? w->rlev[e] : w->clev[e];
}

/*
 * The admissible entry with the lowest Markowitz cost (row count - 1) * (column count - 1).
 * Columns and rows are searched by increasing count; the search stops when no unseen entry can
 * be cheaper. Among equal costs an entry of level 2 wins, then the one found first.
 */
static int find_pivot(work_t *w) {
    plu_t *s = w->s;
    long best_cost = LONG_MAX;
    int best_level = 0, best = -1;
    for (int k = 1; k <= s->m; k++) {
        if (best >= 0 && best_cost <= (long)(k - 1) * (k - 1)) break;
        for (int c = w->cbhead[k]; c >= 0; c = w->cbnext[c]) {
            for (int e = w->chead[c]; e >= 0; e = w->cnext[e]) {
                long cost = (long)(w->rcnt[w->erow[e]] - 1) * (k - 1);
                s->searched++;
                if (cost > best_cost || (cost == best_cost && best_level == 2)) continue;
                int level = entry_level(w, e);
                if (level && (cost < best_cost || level > best_level)) { best_cost = cost; best_level = level; best = e; }
            }
            if (best_cost == 0 && best_level == 2) return best;
        }
        if (best >= 0 && best_cost <= (long)(k - 1) * k) break;
        for (int r = w->rbhead[k]; r >= 0; r = w->rbnext[r]) {
            for (int e = w->rhead[r]; e >= 0; e = w->rnext[e]) {
                long cost = (long)(k - 1) * (w->ccnt[w->ecol[e]] - 1);
                s->searched++;
                if (cost > best_cost || (cost == best_cost && best_level == 2)) continue;
                int level = entry_level(w, e);
                if (level && (cost < best_cost || level > best_level)) { best_cost = cost; best_level = level; best = e; }
            }
            if (best_cost == 0 && best_level == 2) return best;
        }
    }
    return best;
}

static int eliminate(work_t *w, int pe, int k) {
    plu_t *s = w->s;
    int n = w->n, r = w->erow[pe], c = w->ecol[pe];
    double *pivot = SAMPLE(w, pe);
    int unit = is_unit(pivot, n);
    if (!unit && K->recip(pivot, n)) return PLU_NO_PIVOT;
    s->prow[k] = r; s->pcol[k] = c; s->pid[k] = pe;
    /* Row r leaves the active submatrix: its other entries become row k of U. */
    bucket_remove(w->rbhead, w->rbnext, w->rbprev, r, w->rcnt[r]);
    bucket_remove(w->cbhead, w->cbnext, w->cbprev, c, w->ccnt[c]);
    int ustart = s->nu;
    s->uptr[k] = ustart;
    for (int e = w->rhead[r]; e >= 0; e = w->rnext[e]) {
        if (e == pe) continue;
        int j = w->ecol[e];
        s->ucol[s->nu] = j; s->uid[s->nu] = e; s->nu++;
        bucket_remove(w->cbhead, w->cbnext, w->cbprev, j, w->ccnt[j]);
        if (w->cprev[e] >= 0) w->cnext[w->cprev[e]] = w->cnext[e]; else w->chead[j] = w->cnext[e];
        if (w->cnext[e] >= 0) w->cprev[w->cnext[e]] = w->cprev[e];
        w->ccnt[j]--;
        w->cdirty[j] = 1;
    }
    int uend = s->nu, width = uend - ustart;
    /* Column c: every other entry becomes a multiplier (column k of L) and its row is updated. */
    s->lptr[k] = s->nl;
    s->optr[k] = s->nops;
    for (int e = w->chead[c]; e >= 0; e = w->cnext[e]) {
        if (e == pe) continue;
        int i = w->erow[e];
        double *l = SAMPLE(w, e);
        bucket_remove(w->rbhead, w->rbnext, w->rbprev, i, w->rcnt[i]);
        if (w->rprev[e] >= 0) w->rnext[w->rprev[e]] = w->rnext[e]; else w->rhead[i] = w->rnext[e];
        if (w->rnext[e] >= 0) w->rprev[w->rnext[e]] = w->rprev[e];
        w->rcnt[i]--;
        if (!unit) K->mul(l, l, pivot, n);
        s->lrow[s->nl] = i; s->lid[s->nl] = e; s->nl++;
        if (width) {
            if (s->nops + width > s->cap_ops) {
                int room = s->cap_ops * 2 + width;
                int *grown = realloc(s->otarget, sizeof(int) * (size_t)room);
                if (!grown) return PLU_NO_MEMORY;
                s->otarget = grown; s->cap_ops = room;
            }
            for (int t = w->rhead[i]; t >= 0; t = w->rnext[t]) w->pos[w->ecol[t]] = t;
            for (int q = ustart; q < uend; q++) {
                int j = s->ucol[q], t = w->pos[j];
                if (t >= 0) {
                    K->mulsub(SAMPLE(w, t), l, SAMPLE(w, s->uid[q]), n);
                    s->otarget[s->nops++] = t;
                } else {  /* fill-in */
                    if (s->used >= s->cap) return PLU_NO_ROOM;
                    t = s->used++;
                    w->erow[t] = i; w->ecol[t] = j;
                    w->rprev[t] = -1; w->rnext[t] = w->rhead[i];
                    if (w->rhead[i] >= 0) w->rprev[w->rhead[i]] = t;
                    w->rhead[i] = t; w->rcnt[i]++;
                    w->cprev[t] = -1; w->cnext[t] = w->chead[j];
                    if (w->chead[j] >= 0) w->cprev[w->chead[j]] = t;
                    w->chead[j] = t; w->ccnt[j]++;
                    K->negmul(SAMPLE(w, t), l, SAMPLE(w, s->uid[q]), n);
                    s->otarget[s->nops++] = ~t;
                }
            }
            for (int t = w->rhead[i]; t >= 0; t = w->rnext[t]) w->pos[w->ecol[t]] = -1;
        }
        if (w->rcnt[i] == 0) return PLU_NO_PIVOT;  /* an empty row is left: singular */
        w->rdirty[i] = 1;
        bucket_insert(w->rbhead, w->rbnext, w->rbprev, i, w->rcnt[i]);
    }
    for (int q = ustart; q < uend; q++) {
        int j = s->ucol[q];
        if (w->ccnt[j] == 0) return PLU_NO_PIVOT;
        bucket_insert(w->cbhead, w->cbnext, w->cbprev, j, w->ccnt[j]);
    }
    w->rcnt[r] = 0; w->ccnt[c] = 0;
    return PLU_OK;
}

static int analyse(plu_t *s, int nnz, const int32_t *rows, const int32_t *cols, int nsample, const int32_t *sample, double u) {
    int m = s->m, cap = s->cap, F = s->F;
    size_t ints = 6 * (size_t)cap + 4 * (size_t)m + 2 * ((size_t)m + 2) + 4 * (size_t)m + (size_t)m;
    size_t doubles = 2 * (size_t)nsample + 2 * (size_t)cap * nsample;
    void *scratch = malloc(sizeof(double) * doubles + sizeof(int) * ints + 2 * (size_t)m + 2 * (size_t)cap);
    if (!scratch) return PLU_NO_MEMORY;
    work_t w;
    w.s = s; w.n = nsample; w.u2 = u * u;
    w.mx = scratch; w.av = w.mx + 2 * (size_t)nsample;
    int *p = (int *)(w.av + 2 * (size_t)cap * nsample);
    w.erow = p; p += cap; w.ecol = p; p += cap; w.rnext = p; p += cap; w.rprev = p; p += cap; w.cnext = p; p += cap; w.cprev = p; p += cap;
    w.rhead = p; p += m; w.rcnt = p; p += m; w.chead = p; p += m; w.ccnt = p; p += m;
    w.rbhead = p; p += m + 2; w.cbhead = p; p += m + 2;
    w.rbnext = p; p += m; w.rbprev = p; p += m; w.cbnext = p; p += m; w.cbprev = p; p += m;
    w.pos = p; p += m;
    w.rdirty = (unsigned char *)p; w.cdirty = w.rdirty + m; w.rlev = w.cdirty + m; w.clev = w.rlev + cap;
    for (int i = 0; i < m; i++) { w.rhead[i] = w.chead[i] = w.pos[i] = -1; w.rcnt[i] = w.ccnt[i] = 0; }
    for (int i = 0; i < m + 2; i++) w.rbhead[i] = w.cbhead[i] = -1;
    memset(w.rdirty, 1, 2 * (size_t)m);
    int status = PLU_OK;
    /* Link the entries; walking backwards leaves every list in increasing entry order. Entries
     * that are zero at every frequency (not only at the sampled ones) are not part of the pattern. */
    for (int e = nnz - 1; e >= 0; e--) {
        int r = rows[e], c = cols[e];
        if (r < 0 || r >= m || c < 0 || c >= m) { status = PLU_NO_ROOM; break; }
        const double *full = VAL(s, e);
        if (is_zero(full, F)) continue;
        double *a = SAMPLE(&w, e);
        for (int q = 0; q < nsample; q++) { a[2 * q] = full[2 * sample[q]]; a[2 * q + 1] = full[2 * sample[q] + 1]; }
        w.erow[e] = r; w.ecol[e] = c;
        w.rprev[e] = -1; w.rnext[e] = w.rhead[r];
        if (w.rhead[r] >= 0) w.rprev[w.rhead[r]] = e;
        w.rhead[r] = e; w.rcnt[r]++;
        w.cprev[e] = -1; w.cnext[e] = w.chead[c];
        if (w.chead[c] >= 0) w.cprev[w.chead[c]] = e;
        w.chead[c] = e; w.ccnt[c]++;
    }
    for (int i = m - 1; i >= 0 && status == PLU_OK; i--) {
        if (w.rcnt[i] == 0 || w.ccnt[i] == 0) { status = PLU_NO_PIVOT; break; }
        bucket_insert(w.rbhead, w.rbnext, w.rbprev, i, w.rcnt[i]);
        bucket_insert(w.cbhead, w.cbnext, w.cbprev, i, w.ccnt[i]);
    }
    for (int k = 0; k < m && status == PLU_OK; k++) {
        int pe = find_pivot(&w);
        if (pe < 0) { status = PLU_NO_PIVOT; break; }
        status = eliminate(&w, pe, k);
    }
    free(scratch);
    if (status == PLU_OK) { s->lptr[m] = s->nl; s->uptr[m] = s->nu; s->optr[m] = s->nops; }
    return status;
}

/* ------------------------------------------------------------------------------------------
 * Phase 2: numeric factorization of the frequencies f0 .. f0 + n - 1 with the given sequence
 * ---------------------------------------------------------------------------------------- */

static int factor_chunk(plu_t *s, int f0, int n, double u2, double *mx) {
    int m = s->m;
    size_t shift = (size_t)2 * f0;
#define AT(e) (VAL(s, e) + shift)
    for (int k = 0; k < m; k++) {
        double *pivot = AT(s->pid[k]);
        int l0 = s->lptr[k], l1 = s->lptr[k + 1], u0 = s->uptr[k], u1 = s->uptr[k + 1];
        if (l1 > l0 && u1 > u0) {  /* the pivot must be large enough in its row and in its column */
            memset(mx, 0, sizeof(double) * 2 * n);
            K->absmax(mx, pivot, n);
            for (int q = u0; q < u1; q++) K->absmax(mx, AT(s->uid[q]), n);
            if (!K->level(mx, pivot, n, u2)) return k + 1;
            memset(mx, 0, sizeof(double) * 2 * n);
            K->absmax(mx, pivot, n);
            for (int q = l0; q < l1; q++) K->absmax(mx, AT(s->lid[q]), n);
            if (!K->level(mx, pivot, n, u2)) return k + 1;
        }
        if (!is_unit(pivot, n)) {
            if (K->recip(pivot, n)) return k + 1;
            for (int q = l0; q < l1; q++) K->mul(AT(s->lid[q]), AT(s->lid[q]), pivot, n);
            s->punit[k] = 0;
        }
        const int *target = s->otarget + s->optr[k];
        for (int q = l0; q < l1; q++) {
            const double *l = AT(s->lid[q]);
            for (int j = u0; j < u1; j++, target++) {
                if (*target >= 0) K->mulsub(AT(*target), l, AT(s->uid[j]), n);
                else K->negmul(AT(~*target), l, AT(s->uid[j]), n);
            }
        }
    }
#undef AT
    return 0;
}

void plu_free(plu_t *s) {
    if (!s) return;
    free(s->work);
    free(s->otarget);
    free(s->block);
    free(s);
}

/*
 * Factorize. `rows`, `cols` (nnz) give the pattern; entry e has its F values in row e of `val`
 * (cap rows of F complex numbers; rows from `first_free` on are free for fill-in). `val` is
 * overwritten and must stay alive while the factorization is used. The pivots are chosen from
 * `nsample` evenly spread frequencies (0: all of them). `chunk` is the number of frequencies
 * handled together in the numeric phase and in the solves (0: all). Smaller chunks give the same
 * bits; with the values stored entry by entry they were not faster in any measurement.
 *
 * Returns PLU_OK and the handle in *out, or an error code (then *out is NULL and `val` may be
 * partly overwritten). info[0], info[1]: step and first frequency of the chunk where a pivot
 * failed the check (PLU_UNSTABLE).
 */
int plu_factor(plu_t **out, int m, int F, int nnz, const int32_t *rows, const int32_t *cols, double *val, int cap,
               int first_free, double u_select, double u_verify, int nsample, int chunk, int32_t *info) {
    *out = NULL;
    if (simd_level < 0) plu_select(-1);
    if (m <= 0 || F <= 0 || nnz < 0 || cap < nnz || first_free < nnz) return PLU_NO_ROOM;
    if (nsample <= 0 || nsample > F) nsample = F;
    plu_t *s = calloc(1, sizeof(plu_t));
    int32_t *sample = malloc(sizeof(int32_t) * (size_t)nsample);
    if (!s || !sample) { free(s); free(sample); return PLU_NO_MEMORY; }
    for (int q = 0; q < nsample; q++)  /* evenly spread, first and last frequency included */
        sample[q] = nsample == 1 ? F / 2 : (int32_t)(((long)q * (F - 1)) / (nsample - 1));
    s->m = m; s->F = F; s->cap = cap; s->used = first_free; s->val = val;
    size_t ints = (size_t)3 * m + 3 * ((size_t)m + 1) + 4 * (size_t)cap;
    s->block = malloc(ints * sizeof(int) + 4 * (size_t)m);
    s->cap_ops = 8 * cap;
    s->otarget = malloc(sizeof(int) * (size_t)s->cap_ops);
    if (!s->block || !s->otarget) { free(sample); plu_free(s); return PLU_NO_MEMORY; }
    int *p = s->block;
    s->prow = p; p += m; s->pcol = p; p += m; s->pid = p; p += m;
    s->lptr = p; p += m + 1; s->uptr = p; p += m + 1; s->optr = p; p += m + 1;
    s->lrow = p; p += cap; s->lid = p; p += cap; s->ucol = p; p += cap; s->uid = p; p += cap;
    s->punit = (unsigned char *)p; s->flags = s->punit + m;
    struct timespec t0, t1, t2;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    int status = analyse(s, nnz, rows, cols, nsample, sample, u_select);
    free(sample);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    if (status != PLU_OK) { plu_free(s); return status; }
    if (chunk <= 0 || chunk > F) chunk = F;
    s->chunk = chunk;
    double *mx = malloc(sizeof(double) * 2 * (size_t)chunk);
    if (!mx) { plu_free(s); return PLU_NO_MEMORY; }
    memset(s->punit, 1, (size_t)m);
    for (int f0 = 0; f0 < F; f0 += chunk) {
        int failed = factor_chunk(s, f0, f0 + chunk <= F ? chunk : F - f0, u_verify * u_verify, mx);
        if (failed) {
            if (info) { info[0] = failed - 1; info[1] = f0; }
            free(mx); plu_free(s);
            return PLU_UNSTABLE;
        }
    }
    free(mx);
    clock_gettime(CLOCK_MONOTONIC, &t2);
    s->ns_analyse = (t1.tv_sec - t0.tv_sec) * 1000000000L + (t1.tv_nsec - t0.tv_nsec);
    s->ns_numeric = (t2.tv_sec - t1.tv_sec) * 1000000000L + (t2.tv_nsec - t1.tv_nsec);
    *out = s;
    return PLU_OK;
}

/* Counters: entries of L, entries of U above the diagonal, rows of `val` in use, update
 * operations, row/column evaluations and entries looked at by the pivot search, SIMD level,
 * chunk size, pivots that are exactly 1, nanoseconds spent in the analysis and in the numeric phase. */
void plu_stats(const plu_t *s, long *out) {
    long unit = 0;
    for (int k = 0; k < s->m; k++) unit += s->punit[k];
    out[0] = s->nl; out[1] = s->nu; out[2] = s->used; out[3] = s->nops; out[4] = s->evals; out[5] = s->searched;
    out[6] = simd_level; out[7] = s->chunk; out[8] = unit; out[9] = s->ns_analyse; out[10] = s->ns_numeric;
}
/* The pivot sequence: row and column of every step. */
void plu_pivots(const plu_t *s, int32_t *rows, int32_t *cols) {
    for (int k = 0; k < s->m; k++) { rows[k] = s->prow[k]; cols[k] = s->pcol[k]; }
}

/* ------------------------------------------------------------------------------------------
 * Solves. Right-hand sides and solutions are (m, F): the F values of an unknown are contiguous.
 * Rows that are zero at every frequency are skipped (sparse right-hand sides).
 * ---------------------------------------------------------------------------------------- */

static void solve_chunk(const plu_t *s, double *b, double *x, int trans, int f0, int n, unsigned char *bnz, unsigned char *xnz) {
    int m = s->m;
    size_t stride = (size_t)2 * s->F, shift = (size_t)2 * f0, row = sizeof(double) * 2 * (size_t)n;
#define ROW(a, i) ((a) + (size_t)(i) * stride + shift)
#define AT(e) (VAL(s, e) + shift)
    if (!trans) {
        for (int k = 0; k < m; k++) {  /* L y = b */
            int r = s->prow[k];
            if (!bnz[r]) continue;
            for (int q = s->lptr[k]; q < s->lptr[k + 1]; q++) {
                K->mulsub(ROW(b, s->lrow[q]), AT(s->lid[q]), ROW(b, r), n);
                bnz[s->lrow[q]] = 1;
            }
        }
        for (int k = m - 1; k >= 0; k--) {  /* U x = y */
            int r = s->prow[k], c = s->pcol[k], live = bnz[r];
            for (int q = s->uptr[k]; q < s->uptr[k + 1]; q++) {
                if (!xnz[s->ucol[q]]) continue;
                K->mulsub(ROW(b, r), AT(s->uid[q]), ROW(x, s->ucol[q]), n);
                live = 1;
            }
            if (!live) memset(ROW(x, c), 0, row);
            else if (s->punit[k]) memcpy(ROW(x, c), ROW(b, r), row);
            else K->mul(ROW(x, c), ROW(b, r), AT(s->pid[k]), n);
            xnz[c] = (unsigned char)live;
        }
    } else {
        for (int k = 0; k < m; k++) {  /* U^T y = b, y_k kept in x[row of pivot k] */
            int r = s->prow[k], c = s->pcol[k];
            xnz[r] = bnz[c];
            if (!bnz[c]) { memset(ROW(x, r), 0, row); continue; }
            if (s->punit[k]) memcpy(ROW(x, r), ROW(b, c), row);
            else K->mul(ROW(x, r), ROW(b, c), AT(s->pid[k]), n);
            for (int q = s->uptr[k]; q < s->uptr[k + 1]; q++) {
                K->mulsub(ROW(b, s->ucol[q]), AT(s->uid[q]), ROW(x, r), n);
                bnz[s->ucol[q]] = 1;
            }
        }
        for (int k = m - 1; k >= 0; k--) {  /* L^T x = y */
            int r = s->prow[k];
            for (int q = s->lptr[k]; q < s->lptr[k + 1]; q++) {
                if (!xnz[s->lrow[q]]) continue;
                K->mulsub(ROW(x, r), AT(s->lid[q]), ROW(x, s->lrow[q]), n);
                xnz[r] = 1;
            }
        }
    }
#undef ROW
#undef AT
}

/* trans = 0: A x = b, trans = 1: A^T x = b. `b` is overwritten; `x` must not overlap `b`. */
void plu_solve(const plu_t *s, double *b, double *x, int trans) {
    int m = s->m, F = s->F;
    unsigned char *start = s->flags, *bnz = s->flags + m, *xnz = s->flags + 2 * m;
    for (int i = 0; i < m; i++) start[i] = (unsigned char)!is_zero(b + (size_t)i * 2 * F, F);
    for (int f0 = 0; f0 < F; f0 += s->chunk) {
        memcpy(bnz, start, (size_t)m);
        solve_chunk(s, b, x, trans, f0, f0 + s->chunk <= F ? s->chunk : F - f0, bnz, xnz);
    }
}

/*
 * The same for `count` right-hand sides in the layout (F, m, count), as differometor/fast.py
 * keeps them: b[f, i, j]. `x` may be the same array as `b`. Returns nonzero if out of memory.
 */
int plu_solve_fmk(plu_t *s, const double *b, double *x, int count, int trans) {
    int m = s->m, F = s->F;
    size_t size = (size_t)m * F * 2;
    if (!s->work) {
        s->work = malloc(sizeof(double) * 2 * size);
        if (!s->work) return PLU_NO_MEMORY;
    }
    double *wb = s->work, *wx = s->work + size;
    for (int j = 0; j < count; j++) {
        for (int f = 0; f < F; f++) {
            const double *src = b + ((size_t)f * m * count + j) * 2;
            double *dst = wb + (size_t)f * 2;
            for (int i = 0; i < m; i++) {
                dst[(size_t)i * 2 * F] = src[(size_t)i * 2 * count];
                dst[(size_t)i * 2 * F + 1] = src[(size_t)i * 2 * count + 1];
            }
        }
        plu_solve(s, wb, wx, trans);
        for (int f = 0; f < F; f++) {
            double *dst = x + ((size_t)f * m * count + j) * 2;
            const double *src = wx + (size_t)f * 2;
            for (int i = 0; i < m; i++) {
                dst[(size_t)i * 2 * count] = src[(size_t)i * 2 * F];
                dst[(size_t)i * 2 * count + 1] = src[(size_t)i * 2 * F + 1];
            }
        }
    }
    return PLU_OK;
}
