import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def SmartPixLabel(ax, x, y, text=None, color='black', size=14):
    """
    Draw an 'ALEPH' label on a Matplotlib axis at position (x, y), with an optional text.

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
    
    ax.text(x, y, label, transform=ax.transAxes, fontsize=size, color=color, fontweight='bold', style='italic', ha='left', va='bottom')

def SetTicks(ax):
    # Set up ticks
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(axis='both', which='major', labelsize=18, length=8, width=1, direction="in", pad=8)
    ax.tick_params(axis='both', which='minor', labelsize=18, length=4, width=1, direction="in", pad=8)