"""
Safe Python executor — sandboxed code execution for /run, /plot, /calc.

No file I/O · no network · no subprocess · no OS access
Allows: numpy, pandas, matplotlib, scipy, sympy, math, statistics,
        random, json, re, datetime, collections, itertools, seaborn,
        statsmodels, sklearn
"""

import re
import concurrent.futures as _futures
import traceback as _tb
import base64 as _b64
import io as _io_mod
from contextlib import redirect_stdout, redirect_stderr

from graceful.config import EXEC_TIMEOUT

# ── Persistent namespace — variables survive between code cells within a session
code_ns: dict = {}

def reset_code_namespace():
    """Clear the persistent code namespace (called on new session load)."""
    code_ns.clear()
    # Release any open matplotlib figures — prevents memory leak across sessions
    try:
        import matplotlib.pyplot as _plt_reset
        _plt_reset.close('all')
    except Exception:
        pass


def execute_python(code: str, _persist: dict = None) -> tuple[str, str]:
    """
    Execute Python in a hardened sandbox.
    Returns (text_output, html_blob).
    html_blob may contain base64 images and DataFrame tables.
    Pass _persist dict to share variables across calls (persistent namespace).
    """
    if _persist is None:
        _persist = code_ns  # default to session-scoped persistent namespace
    # Modules allowed inside the sandbox — explicit whitelist
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
        'math', 'cmath', 'statistics', 'random', 'json', 're', 'string',
        'textwrap', 'datetime', 'collections', 'itertools', 'functools',
        'operator', 'struct', 'hashlib', 'base64', 'decimal', 'fractions',
        'io', 'abc', 'copy', 'enum', 'typing', 'dataclasses',
        'collections.abc', 'pprint', 'time',
    })

    def _sandboxed():
        import math, cmath, statistics, random, json, re, string, textwrap
        import datetime, collections, itertools, functools, operator
        import struct, hashlib, base64, decimal, fractions, copy, pprint, time
        import io as _io
        import builtins as _builtins_mod

        # ── Safe __import__: allows whitelisted modules, blocks everything else ──
        _real_import = _builtins_mod.__import__
        def _safe_import(name, globs=None, locs=None, fromlist=(), level=0):
            top = name.split('.')[0]
            if top not in _SAFE_MODS and name not in _SAFE_MODS:
                raise ImportError(
                    f"'{name}' is not available in the sandbox. "
                    f"Allowed: numpy, pandas, matplotlib, scipy, math, statistics, ..."
                )
            return _real_import(name, globs, locs, fromlist, level)

        # ── Safe builtins — no open(), no eval/exec, no globals hacks ──
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
            '__import__': _safe_import,   # ← key fix: whitelisted import
            '__build_class__': _builtins_mod.__build_class__,  # needed for class defs
            '__name__': '__main__',
        })

        # Base namespace — stdlib pre-loaded
        _base_ns = {
            '__builtins__': safe_builtins,
            '__name__': '__main__',
            # stdlib — pre-injected so they're available without import too
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

        # ── Merge persistent namespace (user vars from prior cells) ──
        # Persistent vars override base stdlib entries, but builtins stay safe
        ns = dict(_base_ns)
        _unsafe = {'__builtins__', '__name__'}
        for _pk, _pv in _persist.items():
            if _pk not in _unsafe:
                ns[_pk] = _pv

        # ── df alias fallback — if model uses `df` but data is stored under another name ──
        # Inject the first available DataFrame as `df` so model code "just works"
        if 'df' not in ns:
            try:
                import pandas as _pd_check
                _first_df = next(
                    (v for k, v in _persist.items()
                     if isinstance(v, _pd_check.DataFrame) and not k.startswith('_')),
                    None
                )
                if _first_df is not None:
                    ns['df'] = _first_df
            except Exception:
                pass

        # ── Scientific stack ──
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
            np = None

        try:
            import pandas as pd
            ns['pd'] = ns['pandas'] = pd
        except ImportError:
            pd = None

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            if 'matplotlib_style' in dir():
                plt.rcParams.update(matplotlib_style)
            plt.close('all')
            ns['plt'] = plt
            ns['matplotlib'] = matplotlib
        except ImportError:
            plt = None

        try:
            import scipy
            import scipy.stats, scipy.optimize, scipy.signal, scipy.linalg
            import scipy.integrate, scipy.interpolate, scipy.special
            ns['scipy'] = scipy
        except ImportError:
            scipy = None

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
            # Commonly used names pre-imported so model doesn't have to
            for _k in ['symbols','Function','Symbol','solve','simplify','expand',
                       'factor','diff','integrate','limit','series','latex',
                       'sin','cos','tan','exp','log','sqrt','pi','E','I','oo',
                       'Matrix','Eq','Sum','Integral','Derivative','pretty']:
                ns[_k] = locals()[_k]
        except ImportError:
            sympy = None

        try:
            import seaborn as sns
            if 'matplotlib_style' in dir():
                sns.set_theme(style='dark', rc={
                    'figure.facecolor':'#0e0e10','axes.facecolor':'#141416',
                    'text.color':'#c8c8cc','axes.labelcolor':'#c8c8cc',
                })
            ns['sns'] = ns['seaborn'] = sns
        except ImportError:
            sns = None

        try:
            import statsmodels.api as sm
            import statsmodels.formula.api as smf
            ns['sm']  = sm
            ns['smf'] = smf
            ns['statsmodels'] = sm
        except ImportError:
            sm = None

        try:
            import sklearn
            ns['sklearn'] = sklearn
        except ImportError:
            sklearn = None

        stdout_buf = _io_mod.StringIO()
        stderr_buf = _io_mod.StringIO()

        # TRUST BOUNDARY: exec() runs user code in a restricted namespace (no __builtins__).
        # This is a local-only app — the user is running code on their own machine.
        # No filesystem sandbox; code can access files and network if it imports os/socket.
        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(compile(code, '<run>', 'exec'), ns)
        except SystemExit:
            pass
        except Exception:
            stderr_buf.write(_tb.format_exc())

        stdout_val = stdout_buf.getvalue()
        stderr_val = stderr_buf.getvalue()

        # ── Harvest matplotlib figures ──
        fig_html = ""
        if plt is not None:
            try:
                for fig_num in plt.get_fignums():
                    fig = plt.figure(fig_num)
                    if fig.get_axes():
                        buf = _io_mod.BytesIO()
                        fig.savefig(buf, format='png', dpi=130,
                                    bbox_inches='tight', facecolor='#0e0e10')
                        buf.seek(0)
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
                        # Wide tables (>8 cols) get horizontal scroll wrapper
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

        # ── Sync new user-defined variables back to persistent namespace ──
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
        ns_out = {k: v for k, v in ns.items() if k not in _ns_skip and not k.startswith('_')}

        return stdout_val, stderr_val, fig_html, df_html, ns_out

    with _futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_sandboxed)
        try:
            stdout_val, stderr_val, fig_html, df_html, ns_out = fut.result(timeout=EXEC_TIMEOUT)
            # Sync new variables back to persistent namespace (outside the thread)
            _unsafe = {'__builtins__', '__name__'}
            for k, v in ns_out.items():
                if k not in _unsafe:
                    try:
                        _persist[k] = v
                    except Exception:
                        pass
        except _futures.TimeoutError:
            return f"```\n⏱ timed out after {EXEC_TIMEOUT}s\n```", ""
        except Exception as e:
            err_s = str(e)
            if any(x in err_s.lower() for x in ("already borrowed","borrow","metal","gpu","mlx")):
                return "```\n⚠ GPU busy — the model is using Metal resources.\nTry: /run your_code_here  (after the response finishes)\n```", ""
            return f"```\nexecutor error: {e}\n```", ""

    parts = []
    if stdout_val.strip():
        parts.append(f"```\n{stdout_val.rstrip()}\n```")
    if stderr_val.strip():
        err = stderr_val.strip()
        # Clean up sandbox internal noise for readable output
        err = re.sub(r'  File "<string>", line \d+, in _sandboxed\n', '', err)
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
            # Check for bare Name nodes that aren't math names (e.g. "a differential bro")
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
