/* This file was automatically generated by CasADi.
   The CasADi copyright holders make no ownership claim of its contents. */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) vehd_ ## ID
#endif

#include <math.h>
#include <stdio.h>
#include <string.h>
#ifdef MATLAB_MEX_FILE
#include <mex.h>
#endif

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int long long int
#endif

/* Add prefix to internal symbols */
#define casadi_copy CASADI_PREFIX(copy)
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_f1 CASADI_PREFIX(f1)
#define casadi_fill CASADI_PREFIX(fill)
#define casadi_from_mex CASADI_PREFIX(from_mex)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_to_mex CASADI_PREFIX(to_mex)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

void casadi_copy(const casadi_real* x, casadi_int n, casadi_real* y) {
  casadi_int i;
  if (y) {
    if (x) {
      for (i=0; i<n; ++i) *y++ = *x++;
    } else {
      for (i=0; i<n; ++i) *y++ = 0.;
    }
  }
}

void casadi_fill(casadi_real* x, casadi_int n, casadi_real alpha) {
  casadi_int i;
  if (x) {
    for (i=0; i<n; ++i) *x++ = alpha;
  }
}

#ifdef MATLAB_MEX_FILE
casadi_real* casadi_from_mex(const mxArray* p, casadi_real* y, const casadi_int* sp, casadi_real* w) {
  casadi_int nrow, ncol, is_sparse, c, k, p_nrow, p_ncol;
  const casadi_int *colind, *row;
  mwIndex *Jc, *Ir;
  const double* p_data;
  if (!mxIsDouble(p) || mxGetNumberOfDimensions(p)!=2)
    mexErrMsgIdAndTxt("Casadi:RuntimeError",
      "\"from_mex\" failed: Not a two-dimensional matrix of double precision.");
  nrow = *sp++;
  ncol = *sp++;
  colind = sp;
  row = sp+ncol+1;
  p_nrow = mxGetM(p);
  p_ncol = mxGetN(p);
  is_sparse = mxIsSparse(p);
  Jc = 0;
  Ir = 0;
  if (is_sparse) {
    Jc = mxGetJc(p);
    Ir = mxGetIr(p);
  }
  p_data = (const double*)mxGetData(p);
  if (p_nrow==1 && p_ncol==1) {
    casadi_int nnz;
    double v = is_sparse && Jc[1]==0 ? 0 : *p_data;
    nnz = sp[ncol];
    casadi_fill(y, nnz, v);
  } else {
    casadi_int tr = 0;
    if (nrow!=p_nrow || ncol!=p_ncol) {
      tr = nrow==p_ncol && ncol==p_nrow && (nrow==1 || ncol==1);
      if (!tr) mexErrMsgIdAndTxt("Casadi:RuntimeError",
                                 "\"from_mex\" failed: Dimension mismatch. "
                                 "Expected %d-by-%d, got %d-by-%d instead.",
                                 nrow, ncol, p_nrow, p_ncol);
    }
    if (is_sparse) {
      if (tr) {
        for (c=0; c<ncol; ++c)
          for (k=colind[c]; k<colind[c+1]; ++k) w[row[k]+c*nrow]=0;
        for (c=0; c<p_ncol; ++c)
          for (k=Jc[c]; k<(casadi_int) Jc[c+1]; ++k) w[c+Ir[k]*p_ncol] = p_data[k];
        for (c=0; c<ncol; ++c)
          for (k=colind[c]; k<colind[c+1]; ++k) y[k] = w[row[k]+c*nrow];
      } else {
        for (c=0; c<ncol; ++c) {
          for (k=colind[c]; k<colind[c+1]; ++k) w[row[k]]=0;
          for (k=Jc[c]; k<(casadi_int) Jc[c+1]; ++k) w[Ir[k]]=p_data[k];
          for (k=colind[c]; k<colind[c+1]; ++k) y[k]=w[row[k]];
        }
      }
    } else {
      for (c=0; c<ncol; ++c) {
        for (k=colind[c]; k<colind[c+1]; ++k) {
          y[k] = p_data[row[k]+c*nrow];
        }
      }
    }
  }
  return y;
}

#endif

#define casadi_to_double(x) ((double) x)

#ifdef MATLAB_MEX_FILE
mxArray* casadi_to_mex(const casadi_int* sp, const casadi_real* x) {
  casadi_int nrow, ncol, c, k;
#ifndef CASADI_MEX_NO_SPARSE
  casadi_int nnz;
#endif
  const casadi_int *colind, *row;
  mxArray *p;
  double *d;
#ifndef CASADI_MEX_NO_SPARSE
  casadi_int i;
  mwIndex *j;
#endif /* CASADI_MEX_NO_SPARSE */
  nrow = *sp++;
  ncol = *sp++;
  colind = sp;
  row = sp+ncol+1;
#ifndef CASADI_MEX_NO_SPARSE
  nnz = sp[ncol];
  if (nnz!=nrow*ncol) {
    p = mxCreateSparse(nrow, ncol, nnz, mxREAL);
    for (i=0, j=mxGetJc(p); i<=ncol; ++i) *j++ = *colind++;
    for (i=0, j=mxGetIr(p); i<nnz; ++i) *j++ = *row++;
    if (x) {
      d = (double*)mxGetData(p);
      for (i=0; i<nnz; ++i) *d++ = casadi_to_double(*x++);
    }
    return p;
  }
#endif /* CASADI_MEX_NO_SPARSE */
  p = mxCreateDoubleMatrix(nrow, ncol, mxREAL);
  if (x) {
    d = (double*)mxGetData(p);
    for (c=0; c<ncol; ++c) {
      for (k=colind[c]; k<colind[c+1]; ++k) {
        d[row[k]+c*nrow] = casadi_to_double(*x++);
      }
    }
  }
  return p;
}

#endif

#ifndef CASADI_PRINTF
#ifdef MATLAB_MEX_FILE
  #define CASADI_PRINTF mexPrintf
#else
  #define CASADI_PRINTF printf
#endif
#endif

static const casadi_int casadi_s0[9] = {5, 1, 0, 5, 0, 1, 2, 3, 4};
static const casadi_int casadi_s1[5] = {1, 1, 0, 1, 0};

/* f:(i0[5],i1)->(o0[5]) */
static int casadi_f1(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real w0, w1, w2, w3, w4;
  /* #0: @0 = -6.22454 */
  w0 = -6.2245359891353553e+000;
  /* #1: @1 = input[0][0] */
  w1 = arg[0] ? arg[0][0] : 0;
  /* #2: @0 = (@0*@1) */
  w0 *= w1;
  /* #3: @2 = -19.1526 */
  w2 = -1.9152557718424628e+001;
  /* #4: @3 = input[0][2] */
  w3 = arg[0] ? arg[0][2] : 0;
  /* #5: @2 = (@2*@3) */
  w2 *= w3;
  /* #6: @0 = (@0+@2) */
  w0 += w2;
  /* #7: @2 = 4.52694 */
  w2 = 4.5269352648257133e+000;
  /* #8: @4 = input[1][0] */
  w4 = arg[1] ? arg[1][0] : 0;
  /* #9: @2 = (@2*@4) */
  w2 *= w4;
  /* #10: @0 = (@0+@2) */
  w0 += w2;
  /* #11: output[0][0] = @0 */
  if (res[0]) res[0][0] = w0;
  /* #12: output[0][1] = @3 */
  if (res[0]) res[0][1] = w3;
  /* #13: @0 = 0.484347 */
  w0 = 4.8434670116429496e-001;
  /* #14: @0 = (@0*@1) */
  w0 *= w1;
  /* #15: @2 = -7.21475 */
  w2 = -7.2147475614489007e+000;
  /* #16: @2 = (@2*@3) */
  w2 *= w3;
  /* #17: @0 = (@0+@2) */
  w0 += w2;
  /* #18: @2 = 2.99586 */
  w2 = 2.9958602846054334e+000;
  /* #19: @2 = (@2*@4) */
  w2 *= w4;
  /* #20: @0 = (@0+@2) */
  w0 += w2;
  /* #21: output[0][2] = @0 */
  if (res[0]) res[0][2] = w0;
  /* #22: @0 = 20 */
  w0 = 20.;
  /* #23: @2 = input[0][1] */
  w2 = arg[0] ? arg[0][1] : 0;
  /* #24: @4 = cos(@2) */
  w4 = cos( w2 );
  /* #25: @0 = (@0*@4) */
  w0 *= w4;
  /* #26: @4 = sin(@2) */
  w4 = sin( w2 );
  /* #27: @4 = (@1*@4) */
  w4  = (w1*w4);
  /* #28: @0 = (@0-@4) */
  w0 -= w4;
  /* #29: output[0][3] = @0 */
  if (res[0]) res[0][3] = w0;
  /* #30: @0 = 20 */
  w0 = 20.;
  /* #31: @4 = sin(@2) */
  w4 = sin( w2 );
  /* #32: @0 = (@0*@4) */
  w0 *= w4;
  /* #33: @2 = cos(@2) */
  w2 = cos( w2 );
  /* #34: @1 = (@1*@2) */
  w1 *= w2;
  /* #35: @0 = (@0+@1) */
  w0 += w1;
  /* #36: output[0][4] = @0 */
  if (res[0]) res[0][4] = w0;
  return 0;
}

/* F:(i0[5],i1)->(o0[5]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_int i;
  casadi_real **res1=res+1, *rr;
  const casadi_real **arg1=arg+2, *cr, *cs;
  casadi_real *w0=w+5, w1, w2, *w3=w+12, w4, *w5=w+18, *w6=w+23;
  /* #0: @0 = input[0][0] */
  casadi_copy(arg[0], 5, w0);
  /* #1: @1 = 0.00416667 */
  w1 = 4.1666666666666666e-003;
  /* #2: @2 = input[1][0] */
  w2 = arg[1] ? arg[1][0] : 0;
  /* #3: @3 = f(@0, @2) */
  arg1[0]=w0;
  arg1[1]=(&w2);
  res1[0]=w3;
  if (casadi_f1(arg1, res1, iw, w, 0)) return 1;
  /* #4: @4 = 0.0125 */
  w4 = 1.2500000000000001e-002;
  /* #5: @5 = (@4*@3) */
  for (i=0, rr=w5, cs=w3; i<5; ++i) (*rr++)  = (w4*(*cs++));
  /* #6: @5 = (@0+@5) */
  for (i=0, rr=w5, cr=w0, cs=w5; i<5; ++i) (*rr++)  = ((*cr++)+(*cs++));
  /* #7: @6 = f(@5, @2) */
  arg1[0]=w5;
  arg1[1]=(&w2);
  res1[0]=w6;
  if (casadi_f1(arg1, res1, iw, w, 0)) return 1;
  /* #8: @5 = (2.*@6) */
  for (i=0, rr=w5, cs=w6; i<5; ++i) *rr++ = (2.* *cs++ );
  /* #9: @3 = (@3+@5) */
  for (i=0, rr=w3, cs=w5; i<5; ++i) (*rr++) += (*cs++);
  /* #10: @4 = 0.0125 */
  w4 = 1.2500000000000001e-002;
  /* #11: @6 = (@4*@6) */
  for (i=0, rr=w6, cs=w6; i<5; ++i) (*rr++)  = (w4*(*cs++));
  /* #12: @6 = (@0+@6) */
  for (i=0, rr=w6, cr=w0, cs=w6; i<5; ++i) (*rr++)  = ((*cr++)+(*cs++));
  /* #13: @5 = f(@6, @2) */
  arg1[0]=w6;
  arg1[1]=(&w2);
  res1[0]=w5;
  if (casadi_f1(arg1, res1, iw, w, 0)) return 1;
  /* #14: @6 = (2.*@5) */
  for (i=0, rr=w6, cs=w5; i<5; ++i) *rr++ = (2.* *cs++ );
  /* #15: @3 = (@3+@6) */
  for (i=0, rr=w3, cs=w6; i<5; ++i) (*rr++) += (*cs++);
  /* #16: @4 = 0.025 */
  w4 = 2.5000000000000001e-002;
  /* #17: @5 = (@4*@5) */
  for (i=0, rr=w5, cs=w5; i<5; ++i) (*rr++)  = (w4*(*cs++));
  /* #18: @5 = (@0+@5) */
  for (i=0, rr=w5, cr=w0, cs=w5; i<5; ++i) (*rr++)  = ((*cr++)+(*cs++));
  /* #19: @6 = f(@5, @2) */
  arg1[0]=w5;
  arg1[1]=(&w2);
  res1[0]=w6;
  if (casadi_f1(arg1, res1, iw, w, 0)) return 1;
  /* #20: @3 = (@3+@6) */
  for (i=0, rr=w3, cs=w6; i<5; ++i) (*rr++) += (*cs++);
  /* #21: @3 = (@1*@3) */
  for (i=0, rr=w3, cs=w3; i<5; ++i) (*rr++)  = (w1*(*cs++));
  /* #22: @0 = (@0+@3) */
  for (i=0, rr=w0, cs=w3; i<5; ++i) (*rr++) += (*cs++);
  /* #23: @1 = 0.00416667 */
  w1 = 4.1666666666666666e-003;
  /* #24: @3 = f(@0, @2) */
  arg1[0]=w0;
  arg1[1]=(&w2);
  res1[0]=w3;
  if (casadi_f1(arg1, res1, iw, w, 0)) return 1;
  /* #25: @4 = 0.0125 */
  w4 = 1.2500000000000001e-002;
  /* #26: @6 = (@4*@3) */
  for (i=0, rr=w6, cs=w3; i<5; ++i) (*rr++)  = (w4*(*cs++));
  /* #27: @6 = (@0+@6) */
  for (i=0, rr=w6, cr=w0, cs=w6; i<5; ++i) (*rr++)  = ((*cr++)+(*cs++));
  /* #28: @5 = f(@6, @2) */
  arg1[0]=w6;
  arg1[1]=(&w2);
  res1[0]=w5;
  if (casadi_f1(arg1, res1, iw, w, 0)) return 1;
  /* #29: @6 = (2.*@5) */
  for (i=0, rr=w6, cs=w5; i<5; ++i) *rr++ = (2.* *cs++ );
  /* #30: @3 = (@3+@6) */
  for (i=0, rr=w3, cs=w6; i<5; ++i) (*rr++) += (*cs++);
  /* #31: @4 = 0.0125 */
  w4 = 1.2500000000000001e-002;
  /* #32: @5 = (@4*@5) */
  for (i=0, rr=w5, cs=w5; i<5; ++i) (*rr++)  = (w4*(*cs++));
  /* #33: @5 = (@0+@5) */
  for (i=0, rr=w5, cr=w0, cs=w5; i<5; ++i) (*rr++)  = ((*cr++)+(*cs++));
  /* #34: @6 = f(@5, @2) */
  arg1[0]=w5;
  arg1[1]=(&w2);
  res1[0]=w6;
  if (casadi_f1(arg1, res1, iw, w, 0)) return 1;
  /* #35: @5 = (2.*@6) */
  for (i=0, rr=w5, cs=w6; i<5; ++i) *rr++ = (2.* *cs++ );
  /* #36: @3 = (@3+@5) */
  for (i=0, rr=w3, cs=w5; i<5; ++i) (*rr++) += (*cs++);
  /* #37: @4 = 0.025 */
  w4 = 2.5000000000000001e-002;
  /* #38: @6 = (@4*@6) */
  for (i=0, rr=w6, cs=w6; i<5; ++i) (*rr++)  = (w4*(*cs++));
  /* #39: @6 = (@0+@6) */
  for (i=0, rr=w6, cr=w0, cs=w6; i<5; ++i) (*rr++)  = ((*cr++)+(*cs++));
  /* #40: @5 = f(@6, @2) */
  arg1[0]=w6;
  arg1[1]=(&w2);
  res1[0]=w5;
  if (casadi_f1(arg1, res1, iw, w, 0)) return 1;
  /* #41: @3 = (@3+@5) */
  for (i=0, rr=w3, cs=w5; i<5; ++i) (*rr++) += (*cs++);
  /* #42: @3 = (@1*@3) */
  for (i=0, rr=w3, cs=w3; i<5; ++i) (*rr++)  = (w1*(*cs++));
  /* #43: @0 = (@0+@3) */
  for (i=0, rr=w0, cs=w3; i<5; ++i) (*rr++) += (*cs++);
  /* #44: @1 = 0.00416667 */
  w1 = 4.1666666666666666e-003;
  /* #45: @3 = f(@0, @2) */
  arg1[0]=w0;
  arg1[1]=(&w2);
  res1[0]=w3;
  if (casadi_f1(arg1, res1, iw, w, 0)) return 1;
  /* #46: @4 = 0.0125 */
  w4 = 1.2500000000000001e-002;
  /* #47: @5 = (@4*@3) */
  for (i=0, rr=w5, cs=w3; i<5; ++i) (*rr++)  = (w4*(*cs++));
  /* #48: @5 = (@0+@5) */
  for (i=0, rr=w5, cr=w0, cs=w5; i<5; ++i) (*rr++)  = ((*cr++)+(*cs++));
  /* #49: @6 = f(@5, @2) */
  arg1[0]=w5;
  arg1[1]=(&w2);
  res1[0]=w6;
  if (casadi_f1(arg1, res1, iw, w, 0)) return 1;
  /* #50: @5 = (2.*@6) */
  for (i=0, rr=w5, cs=w6; i<5; ++i) *rr++ = (2.* *cs++ );
  /* #51: @3 = (@3+@5) */
  for (i=0, rr=w3, cs=w5; i<5; ++i) (*rr++) += (*cs++);
  /* #52: @4 = 0.0125 */
  w4 = 1.2500000000000001e-002;
  /* #53: @6 = (@4*@6) */
  for (i=0, rr=w6, cs=w6; i<5; ++i) (*rr++)  = (w4*(*cs++));
  /* #54: @6 = (@0+@6) */
  for (i=0, rr=w6, cr=w0, cs=w6; i<5; ++i) (*rr++)  = ((*cr++)+(*cs++));
  /* #55: @5 = f(@6, @2) */
  arg1[0]=w6;
  arg1[1]=(&w2);
  res1[0]=w5;
  if (casadi_f1(arg1, res1, iw, w, 0)) return 1;
  /* #56: @6 = (2.*@5) */
  for (i=0, rr=w6, cs=w5; i<5; ++i) *rr++ = (2.* *cs++ );
  /* #57: @3 = (@3+@6) */
  for (i=0, rr=w3, cs=w6; i<5; ++i) (*rr++) += (*cs++);
  /* #58: @4 = 0.025 */
  w4 = 2.5000000000000001e-002;
  /* #59: @5 = (@4*@5) */
  for (i=0, rr=w5, cs=w5; i<5; ++i) (*rr++)  = (w4*(*cs++));
  /* #60: @5 = (@0+@5) */
  for (i=0, rr=w5, cr=w0, cs=w5; i<5; ++i) (*rr++)  = ((*cr++)+(*cs++));
  /* #61: @6 = f(@5, @2) */
  arg1[0]=w5;
  arg1[1]=(&w2);
  res1[0]=w6;
  if (casadi_f1(arg1, res1, iw, w, 0)) return 1;
  /* #62: @3 = (@3+@6) */
  for (i=0, rr=w3, cs=w6; i<5; ++i) (*rr++) += (*cs++);
  /* #63: @3 = (@1*@3) */
  for (i=0, rr=w3, cs=w3; i<5; ++i) (*rr++)  = (w1*(*cs++));
  /* #64: @0 = (@0+@3) */
  for (i=0, rr=w0, cs=w3; i<5; ++i) (*rr++) += (*cs++);
  /* #65: @1 = 0.00416667 */
  w1 = 4.1666666666666666e-003;
  /* #66: @3 = f(@0, @2) */
  arg1[0]=w0;
  arg1[1]=(&w2);
  res1[0]=w3;
  if (casadi_f1(arg1, res1, iw, w, 0)) return 1;
  /* #67: @4 = 0.0125 */
  w4 = 1.2500000000000001e-002;
  /* #68: @6 = (@4*@3) */
  for (i=0, rr=w6, cs=w3; i<5; ++i) (*rr++)  = (w4*(*cs++));
  /* #69: @6 = (@0+@6) */
  for (i=0, rr=w6, cr=w0, cs=w6; i<5; ++i) (*rr++)  = ((*cr++)+(*cs++));
  /* #70: @5 = f(@6, @2) */
  arg1[0]=w6;
  arg1[1]=(&w2);
  res1[0]=w5;
  if (casadi_f1(arg1, res1, iw, w, 0)) return 1;
  /* #71: @6 = (2.*@5) */
  for (i=0, rr=w6, cs=w5; i<5; ++i) *rr++ = (2.* *cs++ );
  /* #72: @3 = (@3+@6) */
  for (i=0, rr=w3, cs=w6; i<5; ++i) (*rr++) += (*cs++);
  /* #73: @4 = 0.0125 */
  w4 = 1.2500000000000001e-002;
  /* #74: @5 = (@4*@5) */
  for (i=0, rr=w5, cs=w5; i<5; ++i) (*rr++)  = (w4*(*cs++));
  /* #75: @5 = (@0+@5) */
  for (i=0, rr=w5, cr=w0, cs=w5; i<5; ++i) (*rr++)  = ((*cr++)+(*cs++));
  /* #76: @6 = f(@5, @2) */
  arg1[0]=w5;
  arg1[1]=(&w2);
  res1[0]=w6;
  if (casadi_f1(arg1, res1, iw, w, 0)) return 1;
  /* #77: @5 = (2.*@6) */
  for (i=0, rr=w5, cs=w6; i<5; ++i) *rr++ = (2.* *cs++ );
  /* #78: @3 = (@3+@5) */
  for (i=0, rr=w3, cs=w5; i<5; ++i) (*rr++) += (*cs++);
  /* #79: @4 = 0.025 */
  w4 = 2.5000000000000001e-002;
  /* #80: @6 = (@4*@6) */
  for (i=0, rr=w6, cs=w6; i<5; ++i) (*rr++)  = (w4*(*cs++));
  /* #81: @6 = (@0+@6) */
  for (i=0, rr=w6, cr=w0, cs=w6; i<5; ++i) (*rr++)  = ((*cr++)+(*cs++));
  /* #82: @5 = f(@6, @2) */
  arg1[0]=w6;
  arg1[1]=(&w2);
  res1[0]=w5;
  if (casadi_f1(arg1, res1, iw, w, 0)) return 1;
  /* #83: @3 = (@3+@5) */
  for (i=0, rr=w3, cs=w5; i<5; ++i) (*rr++) += (*cs++);
  /* #84: @3 = (@1*@3) */
  for (i=0, rr=w3, cs=w3; i<5; ++i) (*rr++)  = (w1*(*cs++));
  /* #85: @0 = (@0+@3) */
  for (i=0, rr=w0, cs=w3; i<5; ++i) (*rr++) += (*cs++);
  /* #86: output[0][0] = @0 */
  casadi_copy(w0, 5, res[0]);
  return 0;
}

CASADI_SYMBOL_EXPORT int F(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int F_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int F_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void F_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int F_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void F_release(int mem) {
}

CASADI_SYMBOL_EXPORT void F_incref(void) {
}

CASADI_SYMBOL_EXPORT void F_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int F_n_in(void) { return 2;}

CASADI_SYMBOL_EXPORT casadi_int F_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real F_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* F_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* F_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* F_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* F_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int F_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 6;
  if (sz_res) *sz_res = 3;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 28;
  return 0;
}

#ifdef MATLAB_MEX_FILE
void mex_F(int resc, mxArray *resv[], int argc, const mxArray *argv[]) {
  casadi_int i;
  casadi_real w[39];
  casadi_int *iw = 0;
  const casadi_real* arg[6] = {0};
  casadi_real* res[3] = {0};
  if (argc>2) mexErrMsgIdAndTxt("Casadi:RuntimeError","Evaluation of \"F\" failed. Too many input arguments (%d, max 2)", argc);
  if (resc>1) mexErrMsgIdAndTxt("Casadi:RuntimeError","Evaluation of \"F\" failed. Too many output arguments (%d, max 1)", resc);
  if (--argc>=0) arg[0] = casadi_from_mex(argv[0], w, casadi_s0, w+11);
  if (--argc>=0) arg[1] = casadi_from_mex(argv[1], w+5, casadi_s1, w+11);
  --resc;
  res[0] = w+6;
  i = F(arg, res, iw, w+11, 0);
  if (i) mexErrMsgIdAndTxt("Casadi:RuntimeError","Evaluation of \"F\" failed.");
  if (res[0]) resv[0] = casadi_to_mex(casadi_s0, res[0]);
}
#endif

casadi_int main_F(casadi_int argc, char* argv[]) {
  casadi_int j;
  casadi_real* a;
  const casadi_real* r;
  casadi_int flag;
  casadi_int *iw = 0;
  casadi_real w[39];
  const casadi_real* arg[6];
  casadi_real* res[3];
  arg[0] = w+0;
  arg[1] = w+5;
  res[0] = w+6;
  a = w;
  for (j=0; j<6; ++j) if (scanf("%lg", a++)<=0) return 2;
  flag = F(arg, res, iw, w+11, 0);
  if (flag) return flag;
  r = w+6;
  for (j=0; j<5; ++j) CASADI_PRINTF("%g ", *r++);
  CASADI_PRINTF("\n");
  return 0;
}


#ifdef MATLAB_MEX_FILE
void mexFunction(int resc, mxArray *resv[], int argc, const mxArray *argv[]) {
  char buf[2];
  int buf_ok = argc > 0 && !mxGetString(*argv, buf, sizeof(buf));
  if (!buf_ok) {
    mex_F(resc, resv, argc, argv);
    return;
  } else if (strcmp(buf, "F")==0) {
    mex_F(resc, resv, argc-1, argv+1);
    return;
  }
  mexErrMsgTxt("First input should be a command string. Possible values: 'F'");
}
#endif
int main(int argc, char* argv[]) {
  if (argc<2) {
    /* name error */
  } else if (strcmp(argv[1], "F")==0) {
    return main_F(argc-2, argv+2);
  }
  fprintf(stderr, "First input should be a command string. Possible values: 'F'\nNote: you may use function.generate_input to create a command string.\n");
  return 1;
}
#ifdef __cplusplus
} /* extern "C" */
#endif
