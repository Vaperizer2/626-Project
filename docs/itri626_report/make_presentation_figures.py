"""Reformat committed report artwork; never load models, raw images or score data.

Uses matplotlib/numpy from uv.lock and the document tool PyMuPDF (2.1).
Writes report-local vector PDFs: larger gallery labels and numbered confusion axes.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pymupdf as pdf

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / 'figures'
MODELS = ['A_scratch_cnn','B_convnext_tiny','C_siglip_so400m_linear','D_mobileclip2_s3_linear']
LABELS = ['A: scratch CNN','B: fine-tuned ConvNeXt-T','C: frozen SigLIP + linear','D: frozen MobileCLIP2-S3 + linear']


def galleries():
    """Keep each embedded photo and each original title/confidence; only reset labels at 8.5 pt."""
    from matplotlib.font_manager import findfont
    font = str(findfont('DejaVu Sans'))
    for name in MODELS:
        folder = 'figures_four_models' if name.startswith('D_') else 'figures'
        src = pdf.open(ROOT / f'results/itri626/{folder}/fig08_gallery_{name}.pdf')
        page = src[0]
        tiles = sorted([(r, image[0]) for image in page.get_images() for r in page.get_image_rects(image[0])],
                       key=lambda tile: (round(tile[0].y0), tile[0].x0))
        assert len(tiles) == 12, (name, len(tiles))
        out = pdf.open()
        target = out.new_page(width=453.54, height=200)
        target.insert_font(fontname='ReportSans', fontfile=font)
        for i, (r, xref) in enumerate(tiles):
            row, col = divmod(i, 6)
            left = 3 + col * 75.6
            # Original titles sit immediately above their images, within the same column.
            title = page.get_text('text', clip=pdf.Rect(r.x0-3, max(0,r.y0-31), r.x1+3, r.y0)).strip()
            assert title and any(ch.isdigit() for ch in title), (name, i, title)
            result = target.insert_textbox(pdf.Rect(left-1, 2+row*100, left+72, 35+row*100), title,
                                           fontname='ReportSans', fontsize=8.5, lineheight=1.05, align=1)
            assert result >= 0, (name, i, title)
            dest = pdf.Rect(left+4, 35+row*100, left+68, 99+row*100)
            target.insert_image(dest, stream=src.extract_image(xref)['image'])
            if row:
                target.draw_rect(dest, color=(0.698,0.094,0.169), width=1.1)
        out.save(OUT / f'gallery_{name}.pdf', garbage=4, deflate=True)


def confusions():
    data = json.loads((ROOT/'results/itri626/metrics.json').read_text())
    classes = list(data['models'][MODELS[0]]['test']['per_class'])
    assert classes == json.loads((ROOT/'data/v2/splits/split_info.json').read_text())['classes']
    mats = [np.asarray(data['models'][n]['test']['confusion']) for n in MODELS]
    vmax = max(m.max() for m in mats)
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9, 'pdf.fonttype':42})
    for start, tag in [(0,'AB'),(2,'CD')]:
        fig = plt.figure(figsize=(6.3,8.1))
        for row in range(2):
            ax = fig.add_axes([.085, .59-row*.5, .59, .37])
            mat = mats[start+row]
            im = ax.imshow(mat, cmap='Blues', vmin=0, vmax=vmax, interpolation='nearest')
            for i in range(20):
                for j in range(20):
                    if mat[i,j]: ax.text(j,i,str(mat[i,j]),ha='center',va='center',fontsize=8.5,
                                         color='white' if mat[i,j]>vmax*.55 else '#172b3a')
            ax.set_xticks(range(20), range(1,21), fontsize=8, rotation=45, ha='right')
            ax.set_yticks(range(20), range(1,21), fontsize=8)
            ax.tick_params(length=2, pad=2)
            ax.set_xlabel('Predicted class', fontsize=9)
            ax.set_ylabel('True class', fontsize=9)
            ax.set_title(LABELS[start+row],loc='left',fontsize=10,pad=7)
        fig.text(.73,.95,'Class key',fontsize=10,weight='bold')
        for i,c in enumerate(classes):
            name = c.replace('microcontroller board','microcontroller\n    board').replace('power supply module','power supply\n    module')
            fig.text(.73,.91-i*.035,f'{i+1:2}. {name}',fontsize=8.5,va='top')
        cb = fig.colorbar(im,cax=fig.add_axes([.73,.12,.23,.018]), orientation='horizontal')
        cb.set_label('Test images (shared scale)',fontsize=8.5)
        cb.set_ticks([0, 10, 20])
        cb.ax.tick_params(labelsize=8.5)
        fig.savefig(OUT/f'confusion_{tag}.pdf')
        plt.close(fig)


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    galleries()
    confusions()
    print('Wrote six report presentation PDFs from committed artwork and metrics.')
