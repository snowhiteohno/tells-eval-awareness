"""Fill the Apart template with the report. Usage: python paper/build_report.py  -> paper/report.docx
Content lives in paper/content.py as a list of blocks; this file only handles the template plumbing."""
import copy, re, sys
import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

sys.path.insert(0, "paper")
from content import TITLE, AUTHOR, AFFIL, ABSTRACT, BLOCKS

FONT = "Old Standard TT"
d = docx.Document("paper/template.docx")
body = d.element.body


def font(run, size=None, italic=None, bold=None):
    rpr = run._r.get_or_add_rPr()
    f = rpr.find(qn("w:rFonts"))
    if f is None:
        f = rpr.makeelement(qn("w:rFonts"), {}); rpr.insert(0, f)
    for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        f.set(qn(a), FONT)
    if size: run.font.size = Pt(size)
    if italic is not None: run.italic = italic
    if bold is not None: run.bold = bold


def rich(p, text, size=None):
    """**bold**, *italic*, `code` inline markup."""
    for tok in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)", text):
        if not tok:
            continue
        if tok.startswith("**"):
            r = p.add_run(tok[2:-2]); font(r, size, bold=True)
        elif tok.startswith("*"):
            r = p.add_run(tok[1:-1]); font(r, size, italic=True)
        elif tok.startswith("`"):
            r = p.add_run(tok[1:-1]); r.font.name = "Courier New"; r.font.size = Pt(size or 9.5)
        else:
            r = p.add_run(tok); font(r, size)
    return p


def set_par(p, text, size=None, italic=False):
    keep = [r for r in p.runs if r._r.xml.count("footnoteReference")]
    for r in p.runs:
        if r not in keep:
            r._r.getparent().remove(r._r)
    r = p.add_run(text); font(r, size, italic=italic)
    for k in keep:  # keep the sprint footnote marker after the title
        p._p.append(k._r)


# ---- title box ----
box = d.tables[0]
title_p = [p for p in box.rows[0].cells[0].paragraphs if p.style.name == "Title"][0]
set_par(title_p, TITLE)
cell = box.rows[1].cells[0]
tbls = cell._tc.findall(qn("w:tbl"))
authors_tbl, abstract_tbl = tbls[0], tbls[1]
anchor = authors_tbl
for txt, b, it in [(AUTHOR, True, False), (AFFIL, False, True), ("With Apart Research", False, False)]:
    np_ = cell.add_paragraph()
    np_.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = np_.add_run(txt); font(r, 11, italic=it, bold=b)
    anchor.addnext(np_._p); anchor = np_._p
cell._tc.remove(authors_tbl)
ab_cell = docx.table._Cell(abstract_tbl.find(qn("w:tr")).find(qn("w:tc")), cell)
ps = ab_cell.paragraphs
for p in ps[1:]:
    p._p.getparent().remove(p._p)
set_par(ps[0], "")
ps[0].alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
rich(ps[0], ABSTRACT, size=10)

# ---- drop the info box and every guidance paragraph after it ----
info = d.tables[1]._tbl
started = False
for el in list(body.iterchildren()):
    if el is info:
        started = True
    if started and el.tag != qn("w:sectPr"):
        body.remove(el)


# ---- content ----
def caption(text):
    p = d.add_paragraph(); rich(p, text, size=9); p.paragraph_format.space_after = Pt(8)


for kind, *a in BLOCKS:
    if kind == "h2":
        p = d.add_paragraph(style="Heading 2"); r = p.add_run(a[0]); font(r)
    elif kind == "h3":
        p = d.add_paragraph(style="Heading 3"); r = p.add_run(a[0]); font(r)
    elif kind == "p":
        p = rich(d.add_paragraph(), a[0]); p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.space_after = Pt(6)
    elif kind == "bullets":
        for t in a[0]:
            p = rich(d.add_paragraph(), "•  " + t); p.paragraph_format.left_indent = Inches(0.25)
            p.paragraph_format.first_line_indent = Inches(-0.18); p.paragraph_format.space_after = Pt(3)
    elif kind == "fig":
        path, width, cap = a
        p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run().add_picture(path, width=Inches(width)); p.paragraph_format.keep_with_next = True
        caption(cap)
    elif kind == "table":
        cap, header, rows, widths = a
        p = rich(d.add_paragraph(), cap, size=9); p.paragraph_format.keep_with_next = True
        t = d.add_table(rows=1 + len(rows), cols=len(header))
        borders = t._tbl.tblPr.makeelement(qn("w:tblBorders"), {})  # template has no "Table Grid" style
        for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
            borders.append(borders.makeelement(qn(f"w:{side}"), {qn("w:val"): "single", qn("w:sz"): "4", qn("w:color"): "999999"}))
        t._tbl.tblPr.append(borders)
        for i, row in enumerate([header] + rows):
            for j, v in enumerate(row):
                c = t.rows[i].cells[j]; c.paragraphs[0].text = ""
                rr = c.paragraphs[0].add_run(str(v)); font(rr, 8.5, bold=(i == 0))
                if widths: c.width = Inches(widths[j])
        if widths:  # LibreOffice reads the grid, Word reads the cells: set both
            t.autofit = False
            for j, gc in enumerate(t._tbl.tblGrid.findall(qn("w:gridCol"))):
                gc.set(qn("w:w"), str(int(widths[j] * 1440)))
        d.add_paragraph().paragraph_format.space_after = Pt(2)
    elif kind == "pre":
        for ln in a[0].split("\n"):
            p = d.add_paragraph(); r = p.add_run(ln); r.font.name = "Courier New"; r.font.size = Pt(8)
            p.paragraph_format.space_after = Pt(0)
    elif kind == "ref":
        p = rich(d.add_paragraph(), a[0], size=9.5); p.paragraph_format.left_indent = Inches(0.3)
        p.paragraph_format.first_line_indent = Inches(-0.3); p.paragraph_format.space_after = Pt(3)
    elif kind == "pagebreak":
        d.add_page_break()

d.save("paper/report.docx")
print("wrote paper/report.docx")
