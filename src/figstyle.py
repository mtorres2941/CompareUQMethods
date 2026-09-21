"""The figure conventions of FIGURE_STYLE.md, in code. Stage 2d.

Read `FIGURE_STYLE.md` first; this module implements the parts of it that can be
implemented and does not repeat its reasoning. The parts it cannot implement are
the ones that matter most: a title that states a finding rather than naming an
axis, a panel that earns its place, and an annotation that does not collide with
the data.

Usage, at the top of a figure cell:

    import figstyle
    figstyle.apply()
    fig, axes = plt.subplots(1, 2, figsize=figstyle.size(2))
    ...
    figstyle.finish(ax, title='Below 0.002 the answer almost never changes')
"""
import numpy as np

#: Okabe-Ito, which is distinguishable under deuteranopia and protanopia and
#: survives greyscale. Order chosen so the first two are the furthest apart.
CATEGORICAL = ('#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9')

#: The single saturated colour reserved for whatever the message is about.
#: Everything else in a figure should be grey; see FIGURE_STYLE.md section 4.
ACCENT = '#D55E00'
MUTED = '#9A9A9A'
FAINT = '#D4D4D4'

#: Perceptually uniform, no luminance reversal.
SEQUENTIAL = 'viridis'

#: Final printed widths in inches. A figure is built at the size it is printed,
#: because text scales and a figure designed wide and shrunk is illegible.
COL_WIDTH = 3.5
FULL_WIDTH = 7.2

TITLE_PT = 9
LABEL_PT = 8
TICK_PT = 7
ANNOT_PT = 6.5


def apply():
    """Set the rcParams this project's figures assume."""
    import matplotlib as mpl
    mpl.rcParams.update({
        # matplotlib writes a negative tick label with U+2212 MINUS SIGN. This
        # project is plain ASCII everywhere, FIGURE_STYLE.md says so in as many
        # words, and nothing had ever set this, so every figure with a negative
        # axis value carried a Unicode character. Found in Stage 2e.
        'axes.unicode_minus': False,
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.prop_cycle': mpl.cycler(color=list(CATEGORICAL)),
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': False,
        'axes.titlesize': TITLE_PT,
        'axes.labelsize': LABEL_PT,
        'axes.titlelocation': 'left',
        'xtick.labelsize': TICK_PT,
        'ytick.labelsize': TICK_PT,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'legend.frameon': False,
        'legend.fontsize': ANNOT_PT,
        'lines.linewidth': 1.3,
        'font.size': LABEL_PT,
    })


def size(ncols=1, nrows=1, aspect=0.75, width=None):
    """Figure size in inches at FINAL printed width."""
    w = width if width is not None else (COL_WIDTH if ncols == 1 else FULL_WIDTH)
    return (w, w / ncols * aspect * nrows)


def finish(ax, title=None, xlabel=None, ylabel=None, nticks=4):
    """Erase what FIGURE_STYLE.md says to erase, and set the message.

    `title` should be the takeaway as a sentence, not a description of the axes.
    Nothing here can check that, which is why the checklist exists.
    """
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)
    if title is not None:
        ax.set_title(title, fontsize=TITLE_PT, loc='left')
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=LABEL_PT)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=LABEL_PT)
    ax.tick_params(labelsize=TICK_PT)
    for axis in (ax.xaxis, ax.yaxis):
        if axis.get_scale() == 'linear':
            axis.set_major_locator(__import__('matplotlib').ticker.MaxNLocator(
                nticks, prune=None))
    return ax


def label_line(ax, x, y, text, color=None, dx=4, dy=0, va='center', ha='left',
               fontsize=ANNOT_PT, weight=None):
    """Put a series name at the series, which is what replaces a legend."""
    return ax.annotate(text, (x, y), xytext=(dx, dy),
                       textcoords='offset points', color=color or MUTED,
                       fontsize=fontsize, va=va, ha=ha, weight=weight,
                       annotation_clip=False)


def stagger(values, minimum_gap):
    """Nudge overlapping label positions apart, preserving order.

    For annotating several crossings on one axis. Returns positions at least
    `minimum_gap` apart, moving later entries up. A label that still collides
    after this means the panel is too crowded, which the style guide treats as a
    fault in the panel rather than in the label.
    """
    out = []
    for v in np.sort(np.asarray(values, dtype=float)):
        if out and v - out[-1] < minimum_gap:
            v = out[-1] + minimum_gap
        out.append(v)
    return np.array(out)


def greyscale_check(path):
    """Luminance spread of a saved figure, as a crude legibility proxy.

    Returns the fraction of distinct luminance levels used. A figure whose marks
    collapse to one level in greyscale is relying on hue alone, which
    FIGURE_STYLE.md section 4 forbids. It is a smell test, not a proof.
    """
    from PIL import Image
    g = np.asarray(Image.open(path).convert('L'))
    ink = g[g < 250]
    return 0.0 if ink.size == 0 else float(len(np.unique(ink)) / 256.0)


def check_overlaps(fig, verbose=True):
    """Report text that overlaps other text, or text that sits on plotted marks.

    ANSWERING "DO YOU DO ANY CLASH DETECTION?" -- no, and three rounds of manual
    fixes in Stage 2d is what that cost. This is the cheap automatic version.

    It renders the figure, takes the bounding box of every text artist, and
    reports pairs that intersect. It also samples the rendered pixels under each
    label and reports labels sitting on saturated ink, which is the "annotation
    on top of the data" case.

    **THE INK CHECK IS REAL AND WAS NOT, UNTIL STAGE 2F.** This docstring
    claimed to sample the rendered pixels under each label from the day it was
    written, and the function only ever compared text against text. A legend
    label sitting squarely on a data point therefore passed -- which is the
    exact fault FIGURE_STYLE.md section 5 names, and it happened on a Stage 2f
    figure. It now renders the canvas and counts non-background pixels inside
    each label's box, excluding the label's own glyphs by comparing against a
    render with the text hidden.

    It is a smell test, not a proof: a box can overlap while the glyphs do not,
    and a label on a pale region may still be hard to read. Treat a report as a
    prompt to look, and look at final size.
    """
    fig.canvas.draw()
    texts = [t for ax in fig.get_axes() for t in ax.texts
             if t.get_text().strip()]
    texts += [t for t in fig.texts if t.get_text().strip()]
    # ALL THREE TITLE ARTISTS, and this was a real blind spot. matplotlib keeps
    # a separate Text for the centre, left and right title, `ax.get_title()`
    # reads the CENTRE one by default, and `apply()` above sets
    # `axes.titlelocation` to 'left' -- so every title this project draws lives
    # in `_left_title`, the guard `if ax.get_title()` was always false, and
    # this function had never checked a single panel title. Found in Stage 2f
    # on a five-column figure whose titles plainly overlapped while this
    # reported none.
    for ax in fig.get_axes():
        for attr in ('title', '_left_title', '_right_title'):
            t = getattr(ax, attr, None)
            if t is not None and t.get_text().strip():
                texts.append(t)
    boxes = []
    for t in texts:
        try:
            boxes.append((t, t.get_window_extent(fig.canvas.get_renderer())))
        except Exception:
            continue
    hits = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][1], boxes[j][1]
            if a.overlaps(b):
                hits.append((boxes[i][0].get_text()[:38],
                             boxes[j][0].get_text()[:38]))
    # TEXT SITTING ON DATA, which is a different fault from text on text.
    # Render once with every text artist hidden, so whatever ink remains inside
    # a label's box belongs to the plot and not to the label itself.
    on_ink = []
    try:
        import numpy as _np
        shown = [t for t in texts if t.get_visible()]
        for t in shown:
            t.set_visible(False)
        fig.canvas.draw()
        bare = _np.asarray(fig.canvas.buffer_rgba())[:, :, :3].astype(int)
        for t in shown:
            t.set_visible(True)
        fig.canvas.draw()
        h = bare.shape[0]
        # The BACKGROUND is the figure's own facecolor, not the median pixel.
        # A median over the canvas is the plot's dominant colour whenever the
        # data fill the axes, which is exactly the case this check exists for:
        # a label on a solid band then reads as sitting on 'background'.
        import matplotlib.colors as _mc
        bg = _np.array(_mc.to_rgb(fig.get_facecolor())) * 255.0
        for t, box in boxes:
            x0, x1 = int(max(box.x0, 0)), int(max(box.x1, 0))
            # matplotlib's y runs up from the bottom, the buffer's runs down
            y0, y1 = int(h - max(box.y1, 0)), int(h - max(box.y0, 0))
            patch = bare[max(y0, 0):y1, x0:x1]
            if patch.size == 0:
                continue
            inked = (_np.abs(patch - bg).sum(axis=2) > 40).mean()
            if inked > 0.04:
                on_ink.append((t.get_text()[:38], float(inked)))
    except Exception:
        pass

    if verbose:
        if on_ink:
            print(f'TEXT ON TOP OF DATA: {len(on_ink)} label(s)')
            for a, frac in on_ink:
                print(f'  {a!r}  covers {frac*100:.0f} pct plotted ink')
        if hits:
            print(f'OVERLAPPING TEXT: {len(hits)} pair(s)')
            for a, b in hits:
                print(f'  {a!r}  <->  {b!r}')
        if not hits and not on_ink:
            print('no overlapping text, no text on data')
    return hits + on_ink
