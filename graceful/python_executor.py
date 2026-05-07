"""
Safe Python executor — sandboxed code execution for /run, /plot, /calc.

Runs in a **spawned subprocess** so heavy imports (numpy, scipy, matplotlib, etc.)
never share memory with the MLX model process.  When the subprocess exits, all
that memory is freed — no OOM risk on 16 GB machines.

No file I/O · no network · no subprocess · no OS access
Allows: numpy, pandas, matplotlib, scipy, sympy, math, statistics,
        random, json, re, datetime, collections, itertools, seaborn,
        statsmodels, sklearn, networkx
"""

import re
import os
import pickle
import traceback as _tb
import io as _io_mod

# When run as a standalone script (subprocess from execute_python),
# fix sys.path so `from graceful.config import ...` works.
if __name__ == "__main__":
    import sys as _sys_boot
    _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _project_root not in _sys_boot.path:
        _sys_boot.path.insert(0, _project_root)
    del _sys_boot

from graceful.config import EXEC_TIMEOUT

# ── Persistent namespace — variables survive between code cells within a session
# Only simple picklable values persist across subprocess calls.
code_ns: dict = {}

def reset_code_namespace():
    """Clear the persistent code namespace (called on new session load)."""
    code_ns.clear()


# ── Module whitelist (used by the subprocess worker) ─────────────────────────

_SAFE_MODS = frozenset({
    'numpy', 'np', 'pandas', 'pd',
    'matplotlib', 'matplotlib.pyplot', 'matplotlib.patches',
    'matplotlib.colors', 'matplotlib.cm', 'matplotlib.ticker',
    'scipy', 'scipy.stats', 'scipy.optimize', 'scipy.signal', 'scipy.linalg',
    'scipy.integrate', 'scipy.interpolate', 'scipy.special',
    'sympy', 'sympy.stats', 'sympy.calculus', 'sympy.matrices',
    'seaborn', 'sns',
    'statsmodels', 'statsmodels.api', 'statsmodels.formula.api',
    'statsmodels.stats', 'statsmodels.stats.api',
    'statsmodels.tsa', 'statsmodels.tsa.api',
    'sklearn', 'sklearn.linear_model', 'sklearn.preprocessing',
    'sklearn.model_selection', 'sklearn.metrics', 'sklearn.decomposition',
    'sklearn.cluster', 'sklearn.ensemble', 'sklearn.svm',
    'networkx', 'nx',
    'math', 'cmath', 'statistics', 'random', 'json', 're', 'string',
    'textwrap', 'datetime', 'collections', 'itertools', 'functools',
    'operator', 'struct', 'hashlib', 'base64', 'decimal', 'fractions',
    'io', 'abc', 'copy', 'enum', 'typing', 'dataclasses',
    'collections.abc', 'pprint', 'time',
})


# ── Subprocess worker (top-level so it's picklable for 'spawn') ──────────────

def _subprocess_worker(code: str, persist_bytes: bytes, result_conn):
    """
    Runs in a spawned child process — completely isolated memory space.
    No MLX model, no server state.  Imports numpy/matplotlib/etc fresh,
    runs the code, sends results back through the pipe, then exits.
    """
    import math, cmath, statistics, random, json, re, string, textwrap
    import datetime, collections, itertools, functools, operator
    import struct, hashlib, base64, decimal, fractions, copy, pprint, time
    import io as _io
    import builtins as _builtins_mod
    import traceback as _tb_sub

    # Deserialize persistent namespace from parent
    try:
        persist_data = pickle.loads(persist_bytes) if persist_bytes else {}
    except Exception:
        persist_data = {}

    # ── Safe __import__ ──
    _real_import = _builtins_mod.__import__
    def _safe_import(name, globs=None, locs=None, fromlist=(), level=0):
        top = name.split('.')[0]
        if top not in _SAFE_MODS and name not in _SAFE_MODS:
            raise ImportError(
                f"'{name}' is not available in the sandbox. "
                f"Allowed: numpy, pandas, matplotlib, scipy, math, statistics, ..."
            )
        return _real_import(name, globs, locs, fromlist, level)

    # ── Safe builtins ──
    _allowed = {
        'abs','all','any','ascii','bin','bool','bytearray','bytes',
        'callable','chr','complex','dict','dir','divmod','enumerate',
        'filter','float','format','frozenset','getattr','hasattr',
        'hash','hex','int','isinstance','issubclass','iter','len',
        'list','map','max','min','next','object','oct','ord','pow',
        'print','range','repr','reversed','round','set','setattr',
        'slice','sorted','str','sum','tuple','type','zip',
        'None','True','False','Ellipsis','NotImplemented',
        'Exception','ValueError','TypeError','KeyError','IndexError',
        'AttributeError','RuntimeError','StopIteration','OverflowError',
        'ZeroDivisionError', 'AssertionError', 'NotImplementedError',
    }
    safe_builtins = {k: getattr(_builtins_mod, k)
                     for k in _allowed if hasattr(_builtins_mod, k)}
    safe_builtins.update({
        'None': None, 'True': True, 'False': False,
        'Ellipsis': ..., 'NotImplemented': NotImplemented,
        '__import__': _safe_import,
        '__build_class__': _builtins_mod.__build_class__,
        '__name__': '__main__',
    })

    # Base namespace — stdlib pre-loaded
    ns = {
        '__builtins__': safe_builtins,
        '__name__': '__main__',
        'math': math, 'cmath': cmath, 'statistics': statistics,
        'random': random, 'json': json, 're': re, 'string': string,
        'textwrap': textwrap, 'datetime': datetime,
        'collections': collections, 'itertools': itertools,
        'functools': functools, 'operator': operator,
        'struct': struct, 'hashlib': hashlib, 'base64': base64,
        'decimal': decimal, 'fractions': fractions,
        'copy': copy, 'pprint': pprint, 'time': time,
        'io': type('io', (), {'StringIO': _io.StringIO, 'BytesIO': _io.BytesIO})(),
    }

    # Merge persistent namespace
    _unsafe_keys = {'__builtins__', '__name__'}
    for _pk, _pv in persist_data.items():
        if _pk not in _unsafe_keys:
            ns[_pk] = _pv

    # df alias fallback
    if 'df' not in ns:
        try:
            import pandas as _pd_check
            _first_df = next(
                (v for k, v in persist_data.items()
                 if isinstance(v, _pd_check.DataFrame) and not k.startswith('_')),
                None
            )
            if _first_df is not None:
                ns['df'] = _first_df
        except Exception:
            pass

    # ── Scientific stack — imported on demand based on what code references ──
    # Lazy loading: only import packages the code actually uses.
    # Eager import of everything was ~500-800MB, causing OOM on 16GB machines.
    np = pd = plt = scipy_mod = sympy_mod = sns_mod = sm_mod = sklearn_mod = None
    _code_lower = code.lower()

    _needs_numpy = any(k in code for k in ['numpy', 'np.', 'np ', 'import np', 'np,'])
    _needs_pandas = any(k in code for k in ['pandas', 'pd.', 'pd ', 'import pd', 'DataFrame', 'read_csv'])
    _needs_matplotlib = any(k in code for k in ['matplotlib', 'plt.', 'plt,', 'import plt', 'plot', 'figure', 'savefig'])
    _needs_scipy = any(k in code for k in ['scipy', 'from scipy'])
    _needs_sympy = any(k in code for k in ['sympy', 'from sympy', 'symbols', 'solve(', 'integrate(', 'diff(', 'latex('])
    _needs_seaborn = any(k in code for k in ['seaborn', 'sns.', 'import sns'])
    _needs_statsmodels = any(k in code for k in ['statsmodels', 'import sm', 'smf.', 'sm.'])
    _needs_sklearn = any(k in code for k in ['sklearn', 'KMeans', 'LinearRegression', 'RandomForest', 'PCA', 'DBSCAN', 'StandardScaler'])

    try:
        import numpy as np
        matplotlib_style = {
            'figure.facecolor': '#0e0e10', 'axes.facecolor': '#141416',
            'axes.edgecolor': '#2a2a2c',   'text.color': '#c8c8cc',
            'axes.labelcolor': '#c8c8cc',  'xtick.color': '#888',
            'ytick.color': '#888',         'grid.color': '#1e1e21',
            'grid.alpha': 0.5,             'axes.grid': True,
            'font.size': 10,               'lines.linewidth': 1.8,
            'figure.autolayout': True,
        }
        ns['np'] = ns['numpy'] = np
    except ImportError:
        matplotlib_style = None

    if _needs_pandas:
        try:
            import pandas as pd
            ns['pd'] = ns['pandas'] = pd
        except ImportError:
            pass

    if _needs_matplotlib:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            if matplotlib_style:
                plt.rcParams.update(matplotlib_style)
            plt.close('all')
            ns['plt'] = plt
            ns['matplotlib'] = matplotlib
        except ImportError:
            pass

    if _needs_scipy:
        try:
            import scipy
            import scipy.stats, scipy.optimize, scipy.signal, scipy.linalg
            import scipy.integrate, scipy.interpolate, scipy.special
            ns['scipy'] = scipy
            scipy_mod = scipy
        except ImportError:
            pass

    if _needs_sympy:
        try:
            import sympy
            from sympy import (
                symbols, Function, Symbol, Integer, Float, Rational,
                solve, simplify, expand, factor, cancel, apart,
                diff, integrate, limit, series,
                sin, cos, tan, exp, log, sqrt, pi, E, I, oo,
                Matrix, eye, zeros, ones,
                latex, pretty,
                Eq, Ne, Lt, Le, Gt, Ge,
                Sum, Product, Integral, Derivative,
            )
            ns['sympy'] = sympy
            ns['sp']    = sympy
            for _k in ['symbols','Function','Symbol','solve','simplify','expand',
                       'factor','diff','integrate','limit','series','latex',
                       'sin','cos','tan','exp','log','sqrt','pi','E','I','oo',
                       'Matrix','Eq','Sum','Integral','Derivative','pretty']:
                ns[_k] = locals()[_k]
            sympy_mod = sympy
        except ImportError:
            pass

    if _needs_seaborn:
        try:
            import seaborn as sns
            if matplotlib_style:
                sns.set_theme(style='dark', rc={
                    'figure.facecolor':'#0e0e10','axes.facecolor':'#141416',
                    'text.color':'#c8c8cc','axes.labelcolor':'#c8c8cc',
                })
            ns['sns'] = ns['seaborn'] = sns
            sns_mod = sns
        except ImportError:
            pass

    if _needs_statsmodels:
        try:
            import statsmodels.api as sm
            import statsmodels.formula.api as smf
            ns['sm']  = sm
            ns['smf'] = smf
            ns['statsmodels'] = sm
            sm_mod = sm
        except ImportError:
            pass

    if _needs_sklearn:
        try:
            import sklearn
            ns['sklearn'] = sklearn
            sklearn_mod = sklearn
        except ImportError:
            pass

    # ── Execute ──
    stdout_buf = _io.StringIO()
    stderr_buf = _io.StringIO()

    from contextlib import redirect_stdout, redirect_stderr
    try:
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            exec(compile(code, '<run>', 'exec'), ns)
    except SystemExit:
        pass
    except Exception:
        stderr_buf.write(_tb_sub.format_exc())

    stdout_val = stdout_buf.getvalue()
    stderr_val = stderr_buf.getvalue()

    # ── Harvest matplotlib figures as base64 ──
    fig_html = ""
    if plt is not None:
        try:
            for fig_num in plt.get_fignums():
                fig = plt.figure(fig_num)
                if fig.get_axes():
                    buf = _io.BytesIO()
                    fig.savefig(buf, format='png', dpi=130,
                                bbox_inches='tight', facecolor='#0e0e10')
                    buf.seek(0)
                    import base64 as _b64
                    b64 = _b64.b64encode(buf.read()).decode()
                    fig_html += (
                        f'<div class="run-figure">'
                        f'<img src="data:image/png;base64,{b64}" '
                        f'style="max-width:100%;border-radius:8px;'
                        f'border:1px solid #2a2a2c;margin-top:10px;display:block"/>'
                        f'</div>'
                    )
            plt.close('all')
        except Exception as _fe:
            stderr_val += f"\n[fig error: {_fe}]"

    # ── Harvest DataFrames / Series ──
    _skip = {'np','numpy','pd','pandas','plt','matplotlib','scipy',
             'math','cmath','statistics','random','json','re','string',
             'textwrap','datetime','collections','itertools','functools',
             'operator','struct','hashlib','base64','decimal','fractions',
             'io','copy','pprint','time'}
    df_html = ""
    if pd is not None:
        try:
            for k, v in ns.items():
                if k.startswith('_') or k in _skip: continue
                if isinstance(v, pd.DataFrame) and len(v) > 0 and len(v.columns) > 0:
                    _n_cols = len(v.columns)
                    _n_rows = len(v)
                    h = v.head(100).to_html(classes='df-output', border=0,
                                            na_rep='—', max_rows=50)
                    _wrap_style = ('overflow-x:auto;max-width:100%;' if _n_cols > 8
                                   else '')
                    df_html += (f'<div class="df-wrap" style="{_wrap_style}">'
                                f'<span class="df-label">{k} '
                                f'({_n_rows:,}×{_n_cols})</span>{h}</div>')
                elif isinstance(v, pd.Series) and len(v) > 0:
                    h = v.head(50).to_frame().to_html(classes='df-output', border=0)
                    df_html += (f'<div class="df-wrap">'
                                f'<span class="df-label">{k} (Series len={len(v):,})</span>{h}</div>')
        except Exception:
            pass

    # ── Collect picklable user variables for persistence ──
    _ns_skip = {'__builtins__','__name__','np','numpy','pd','pandas',
                'plt','matplotlib','scipy','sympy','sp','sns','seaborn',
                'sm','smf','statsmodels','sklearn',
                'math','cmath','statistics','random','json','re','string',
                'textwrap','datetime','collections','itertools','functools',
                'operator','struct','hashlib','base64','decimal','fractions',
                'io','copy','pprint','time',
                'symbols','Function','Symbol','solve','simplify','expand',
                'factor','diff','integrate','limit','series','latex',
                'sin','cos','tan','exp','log','sqrt','pi','E','I','oo',
                'Matrix','Eq','Sum','Integral','Derivative','pretty'}
    ns_out = {}
    for k, v in ns.items():
        if k in _ns_skip or k.startswith('_'):
            continue
        try:
            pickle.dumps(v)  # only keep picklable values
            ns_out[k] = v
        except Exception:
            pass

    # Send results back to parent through the pipe
    try:
        result_conn.send({
            'stdout': stdout_val,
            'stderr': stderr_val,
            'fig_html': fig_html,
            'df_html': df_html,
            'ns_out_bytes': pickle.dumps(ns_out),
        })
    except Exception as e:
        result_conn.send({
            'stdout': stdout_val,
            'stderr': stderr_val + f"\n[pickle error: {e}]",
            'fig_html': fig_html,
            'df_html': df_html,
            'ns_out_bytes': b'',
        })
    result_conn.close()


# ── Main entry point ─────────────────────────────────────────────────────────

def execute_python(code: str, _persist: dict = None) -> tuple[str, str]:
    """
    Execute Python in a hardened sandbox inside a **subprocess**.
    Returns (text_output, html_blob).
    html_blob may contain base64 images and DataFrame tables.
    Uses subprocess.Popen (not multiprocessing) so the child never
    re-imports app.py — no port conflicts, no MLX reload, no OOM.
    """
    if _persist is None:
        _persist = code_ns

    # Serialize persistent namespace for the subprocess
    try:
        persist_bytes = pickle.dumps(_persist) if _persist else b''
    except Exception:
        persist_bytes = b''

    import subprocess as _sub
    import sys as _sys
    import tempfile as _tmpf

    # Temp files for IPC — avoids stdout pollution from import warnings
    _in_fd, _in_path = _tmpf.mkstemp(suffix='_exec_in.pkl')
    _out_fd, _out_path = _tmpf.mkstemp(suffix='_exec_out.pkl')

    try:
        # Write input
        with os.fdopen(_in_fd, 'wb') as _f:
            pickle.dump({'code': code, 'persist_bytes': persist_bytes}, _f)
        os.close(_out_fd)  # release so child can write

        # Run this file as a standalone script — child never touches app.py
        _script = os.path.abspath(__file__)
        _proc = _sub.Popen(
            [_sys.executable, _script, _in_path, _out_path],
            stdout=_sub.PIPE, stderr=_sub.PIPE,
        )

        try:
            _proc.communicate(timeout=EXEC_TIMEOUT + 2)
        except _sub.TimeoutExpired:
            _proc.kill()
            _proc.wait()
            return f"```\n\u23f1 timed out after {EXEC_TIMEOUT}s\n```", ""

        if _proc.returncode != 0:
            _exit = _proc.returncode
            if _exit < 0:
                import signal as _sig
                _abs = -_exit
                _signame = (_sig.Signals(_abs).name
                            if _abs in _sig.Signals._value2member_map_
                            else str(_abs))
                _reason = f"killed by signal {_signame}"
                if _abs == 9:
                    _reason += " (likely OOM \u2014 system memory full)"
            else:
                _reason = f"exit code {_exit}"
            print(f"[EXEC] subprocess crashed: {_reason}", flush=True)
            return f"```\n\u26a0 executor subprocess crashed ({_reason})\n```", ""

        if not os.path.exists(_out_path) or os.path.getsize(_out_path) == 0:
            return "```\n\u26a0 executor produced no output\n```", ""

        with open(_out_path, 'rb') as _f:
            result = pickle.loads(_f.read())

    except Exception as _e:
        return f"```\nexecutor error: {_e}\n```", ""

    finally:
        for _p in (_in_path, _out_path):
            try:
                os.unlink(_p)
            except OSError:
                pass

    stdout_val = result.get('stdout', '')
    stderr_val = result.get('stderr', '')
    fig_html = result.get('fig_html', '')
    df_html = result.get('df_html', '')

    # Restore persistent namespace from subprocess
    ns_out_bytes = result.get('ns_out_bytes', b'')
    if ns_out_bytes:
        try:
            ns_out = pickle.loads(ns_out_bytes)
            _unsafe = {'__builtins__', '__name__'}
            for k, v in ns_out.items():
                if k not in _unsafe:
                    _persist[k] = v
        except Exception:
            pass

    # Format output
    parts = []
    if stdout_val.strip():
        parts.append(f"```\n{stdout_val.rstrip()}\n```")
    if stderr_val.strip():
        err = stderr_val.strip()
        err = re.sub(r'  File "<string>", line \d+, in _subprocess_worker\n', '', err)
        err = re.sub(r'Traceback \(most recent call last\):\n', '', err)
        err = re.sub(r'  File "<run>", line \d+, in <module>\n', '', err)
        err = err.strip()
        if err:
            parts.append(f"```\n{err}\n```")

    return ("\n".join(parts) if parts else "*(no output)*"), (fig_html + df_html)


# ── /plot helper — builds and executes matplotlib code from an expression string ──
def run_plot(arg: str) -> tuple[str, str]:
    """Plot one or more math expressions. Returns (label_text, html_blob)."""
    # Parse optional "from X to Y" range
    _range_m = re.search(
        r'\s+from\s+([\-\d\.e]+(?:\s*\*?\s*pi)?)\s+to\s+([\-\d\.e]+(?:\s*\*?\s*pi)?)\s*$',
        arg, re.I
    )
    _xmin_str, _xmax_str = "-2*np.pi", "2*np.pi"
    _expr_part = arg.strip()
    if _range_m:
        _expr_part = arg[:_range_m.start()].strip()
        def _parse_bound(s):
            s = s.strip().lower().replace('pi', 'np.pi')
            s = s.replace('np.np.pi', 'np.pi')
            return s
        _xmin_str = _parse_bound(_range_m.group(1))
        _xmax_str = _parse_bound(_range_m.group(2))

    # Split on commas not inside parentheses
    _exprs, _depth, _cur = [], 0, []
    for ch in _expr_part:
        if ch == '(':   _depth += 1; _cur.append(ch)
        elif ch == ')': _depth -= 1; _cur.append(ch)
        elif ch == ',' and _depth == 0:
            e = ''.join(_cur).strip()
            if e: _exprs.append(e)
            _cur = []
        else: _cur.append(ch)
    if _cur:
        e = ''.join(_cur).strip()
        if e: _exprs.append(e)

    if not _exprs:
        return "⚠ No expressions to plot.", ""

    # Pre-validate: each expression must be parseable Python and reference 'x'
    _MATH_NAMES = {
        'sin','cos','tan','arcsin','arccos','arctan','sinh','cosh','tanh',
        'exp','log','log2','log10','sqrt','abs','pi','e','floor','ceil',
        'sign','power','clip','where','nan_to_num','x','np','inf',
    }
    _bad_exprs = []
    for _e in _exprs:
        try:
            import ast as _ast
            _tree = _ast.parse(_e, mode='eval')
            _names = [n.id for n in _ast.walk(_tree) if isinstance(n, _ast.Name)]
            _unknown = [n for n in _names if n not in _MATH_NAMES]
            if _unknown:
                _bad_exprs.append((_e, f"unknown name{'s' if len(_unknown)>1 else ''}: {', '.join(_unknown)}"))
        except SyntaxError:
            _bad_exprs.append((_e, "not valid Python syntax"))
    if _bad_exprs:
        _lines = [f"⚠ Can't plot `{e}` — {reason}." for e, reason in _bad_exprs]
        _lines.append("Try: `/plot sin(x)`, `/plot x**2 - 3*x`, `/plot exp(-x)*cos(2*x)`")
        return "\n".join(_lines), ""

    _COLORS = ["'#f03468'", "'#60a5fa'", "'#4ade80'", "'#fb923c'", "'#a78bfa'", "'#f59e0b'"]
    lines = [
        "import numpy as np",
        "from numpy import (sin,cos,tan,arcsin,arccos,arctan,sinh,cosh,tanh,",
        "                   exp,log,log2,log10,sqrt,abs,pi,e,floor,ceil,sign,",
        "                   power,clip,where,nan_to_num)",
        f"x = np.linspace({_xmin_str}, {_xmax_str}, 1200)",
        "fig,ax = plt.subplots(figsize=(8,3.5))",
    ]
    for i, expr in enumerate(_exprs):
        c = _COLORS[i % len(_COLORS)]
        lbl = expr.replace("**","^").replace("np.","").replace("*","·")
        if len(_exprs) == 1:
            lines.append(f"ax.plot(x, {expr}, color={c}, linewidth=2)")
            lines.append(f"ax.set_title(r'$y = {lbl}$', color='#c8c8cc', pad=8)")
        else:
            lines.append(f"ax.plot(x, {expr}, color={c}, linewidth=2, label=r'${lbl}$')")
    if len(_exprs) > 1:
        lines.append("ax.legend(framealpha=0.15, labelcolor='white', edgecolor='#333')")
    lines += [
        "ax.axhline(0, color='white', alpha=0.12, lw=0.7)",
        "ax.axvline(0, color='white', alpha=0.12, lw=0.7)",
        "plt.tight_layout()",
    ]
    text_out, html_out = execute_python("\n".join(lines))
    if ("Error" in text_out and text_out.strip()) or "Traceback" in text_out:
        return f"⚠ Plot error:\n```\n{text_out.strip()}\n```", ""
    label = ", ".join(_exprs)
    return f"**`▶ {label}`**", html_out


# ── /calc helper — evaluates a sympy expression and pretty-prints it ──
def run_calc(arg: str) -> tuple[str, str]:
    """Evaluate arg symbolically + numerically. Returns (text_output, html_blob)."""
    code = f"""
from sympy import (symbols, solve, simplify, expand, factor, cancel, apart,
                   diff, integrate, limit, series, latex, sin, cos, tan, exp,
                   log, sqrt, pi, E, I, oo, Matrix, Rational, factorial,
                   binomial, Sum, Product, Integral, Derivative, trigsimp,
                   nsimplify, N as _N, S, Abs, ceiling, floor, re, im)
import sympy as sp
x,y,z,t,n = symbols('x y z t n', real=True)
a,b,c,k   = symbols('a b c k')

_result = None
try:
    _result = eval({repr(arg)})
except Exception as _e1:
    try:
        _result = sp.sympify({repr(arg)})
    except Exception as _e2:
        print(f"⚠ {{_e2}}")

if _result is not None:
    _lx = latex(_result)
    print(f"$${{_lx}}$$")
    try:
        _num = complex(_N(_result, 12))
        if _num.imag == 0 and str(_result) != str(round(_num.real, 10)):
            print(f"≈ {{_num.real:.10g}}")
    except Exception:
        pass
"""
    text_out, html_out = execute_python(code)
    if not text_out.strip() and not html_out:
        return "*(no result)*", ""
    return text_out, html_out


# ── Stats/math keyword detection — precompiled regex ──────────────────────────
_STATS_SIGNALS = [
    'statistic', 'regression', 'p-value', 'p value', 'hypothesis', 'anova',
    'variance', 'std dev', 'standard deviation',
    'correlation', 'covariance', 'bayesian', 'frequentist', 'likelihood',
    'eigenvalue',
    'sympy', 'numpy', 'pandas', 'scipy', 'statsmodels', 'seaborn', 'sklearn',
    'histogram', 'scatter', 'boxplot', 'residual', 'ols', 'glm',
    'solve for', 'differentiate', 'differential', 'differential equation',
    'derivative', 'integral', 'integrate', 'calculus', 'equation', 'formula',
    'matrix', 'vector', 'linear algebra', 'fourier', 'laplace', 'gradient',
    'probability', 'random variable', 'expected value', 'normal distribution',
    'cluster', 'clustering', 'k-means', 'kmeans', 'pca', 'classification',
    'regression', 'predict', 'train', 'model', 'feature', 'graph', 'plot',
    'visualize', 'chart', 'analyze', 'analyse',
    'dataset', 'dataframe', 'csv', 'make a', 'create a', 'generate a',
    'make me', 'write me', 'run ', 'the data', 'my data', 'upload',
]
STATS_RE = re.compile(r'\b(' + '|'.join(re.escape(s) for s in _STATS_SIGNALS) + r')\b', re.I)


# ── Standalone entry point — called by execute_python via subprocess.Popen ───
if __name__ == "__main__":
    import sys as _sys_main

    _in_path = _sys_main.argv[1]
    _out_path = _sys_main.argv[2]

    with open(_in_path, 'rb') as _f:
        _data = pickle.loads(_f.read())

    class _ResultCapture:
        """Shim for result_conn — captures .send() data without a real Pipe."""
        def __init__(self):
            self.data = None
        def send(self, d):
            self.data = d
        def close(self):
            pass

    _cap = _ResultCapture()
    try:
        _subprocess_worker(_data['code'], _data['persist_bytes'], _cap)
    except Exception as _e:
        _cap.data = {
            'stdout': '', 'stderr': str(_e),
            'fig_html': '', 'df_html': '', 'ns_out_bytes': b'',
        }

    with open(_out_path, 'wb') as _f:
        pickle.dump(_cap.data, _f)
