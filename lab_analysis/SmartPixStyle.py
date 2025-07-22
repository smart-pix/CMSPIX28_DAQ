import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def SmartPixLabel(ax, x, y, text=None, color='black', size=14, fontweight='bold', style='italic'):
    """
    Draw a 'SmartPixel' label on a Matplotlib axis at position (x, y), with an optional text.

    Parameters:
    - ax: Matplotlib axis object
    - x, y: Position in axis coordinates (0-1)
    - text: Optional string to appear next to "ALEPH"
    - color: Matplotlib color (default is black)
    - size: Font size (default is 14)
    """
    label = "SmartPixels"
    if text:
        label += f" {text}"
    
    ax.text(x, y, label, transform=ax.transAxes, fontsize=size, color=color, fontweight=fontweight, style=style, ha='left', va='bottom')
    # ax.text(x, y, label, transform=ax.transAxes, fontsize=size, color=color, ha='left', va='bottom')

def SetTicks(ax, nbinsMajor=5, nbinsMinor=None):
    # Set up ticks
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=nbinsMajor))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=nbinsMajor))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(n=nbinsMinor))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(n=nbinsMinor))
    ax.tick_params(axis='both', which='major', labelsize=18, length=8, width=1, direction="in", pad=8)
    ax.tick_params(axis='both', which='minor', labelsize=18, length=4, width=1, direction="in", pad=8)
