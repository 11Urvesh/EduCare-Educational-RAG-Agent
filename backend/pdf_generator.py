import unicodedata

from fpdf import FPDF

INDENT = 8  # mm indent for MCQ options


def _safe(text: str) -> str:
    """Convert any Unicode text to safe ASCII for Helvetica font."""
    replacements = {
        "\u2014": "-",   # em dash
        "\u2013": "-",   # en dash
        "\u2012": "-",   # figure dash
        "\u2015": "-",   # horizontal bar
        "\u2212": "-",   # minus sign
        "\u2018": "'",   # left single quote
        "\u2019": "'",   # right single quote
        "\u201c": '"',   # left double quote
        "\u201d": '"',   # right double quote
        "\u2026": "...", # ellipsis
        "\u2022": "*",   # bullet
        "\u00b7": "*",   # middle dot
        "\u00a0": " ",   # non-breaking space
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    text = unicodedata.normalize("NFKD", text)
    return text.encode("ascii", errors="ignore").decode("ascii")


class _PDF(FPDF):
    def __init__(self, title: str):
        super().__init__()
        self._doc_title = _safe(title)

    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, self._doc_title, align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    @property
    def usable_width(self) -> float:
        """Full usable width between left and right margins."""
        return self.w - self.l_margin - self.r_margin


def _meta_block(pdf: _PDF, meta: dict):
    pdf.set_font("Helvetica", "", 10)
    parts = [
        f"Type: {meta['type'].upper()}",
        f"Total Marks: {meta['total_marks']}",
        f"Duration: {meta['duration_minutes']} min",
        f"Difficulty: {meta['difficulty'].capitalize()}",
    ]
    if meta.get("topic_focus") and meta["topic_focus"] != "General":
        parts.append(f"Topic: {_safe(meta['topic_focus'])}")
    pdf.cell(pdf.usable_width, 8, _safe("  |  ".join(parts)), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_draw_color(180, 180, 180)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(6)


def generate_mcq_papers(questions: list, meta: dict, marks_per_q: int) -> tuple[bytes, bytes]:
    """Returns (question_paper_bytes, answer_key_bytes)."""

    # -- Question Paper --
    qp = _PDF("EduCare - MCQ Question Paper")
    qp.add_page()
    qp.set_auto_page_break(auto=True, margin=15)
    _meta_block(qp, meta)

    W = qp.usable_width

    qp.set_font("Helvetica", "B", 10)
    qp.cell(W, 8, "Instructions: Choose the correct option for each question.", new_x="LMARGIN", new_y="NEXT")
    qp.ln(4)

    for q in questions:
        qp.set_font("Helvetica", "B", 10)
        qp.multi_cell(W, 7, _safe(f"Q{q['q_no']}. {q['question']}  [{marks_per_q} mark(s)]"))
        qp.set_font("Helvetica", "", 10)
        for opt, text in q["options"].items():
            qp.set_x(qp.l_margin + INDENT)
            qp.multi_cell(W - INDENT, 6, _safe(f"{opt}.  {text}"))
        qp.ln(4)

    # -- Answer Key --
    ak = _PDF("EduCare - MCQ Answer Key")
    ak.add_page()
    ak.set_auto_page_break(auto=True, margin=15)
    _meta_block(ak, meta)

    W = ak.usable_width

    for q in questions:
        ak.set_font("Helvetica", "B", 10)
        ak.set_x(ak.l_margin)
        ak.multi_cell(W, 7, _safe(f"Q{q['q_no']}. {q['question']}"))
        ak.set_font("Helvetica", "", 10)
        ak.set_fill_color(220, 240, 220)
        ak.set_x(ak.l_margin)
        correct_letter = q['correct_answer']
        correct_text = q.get('options', {}).get(correct_letter, "")
        answer_line = f"  Answer: {correct_letter}. {correct_text}" if correct_text else f"  Answer: {correct_letter}"
        ak.multi_cell(W, 6, _safe(answer_line), fill=True)
        if q.get("explanation"):
            ak.set_font("Helvetica", "I", 9)
            ak.set_x(ak.l_margin)
            ak.multi_cell(W, 6, _safe(f"  Explanation: {q['explanation']}"))
        ak.ln(4)

    return bytes(qp.output()), bytes(ak.output())


def generate_descriptive_papers(questions: list, meta: dict) -> tuple[bytes, bytes]:
    """Returns (question_paper_bytes, answer_key_bytes)."""

    # -- Question Paper --
    qp = _PDF("EduCare - Descriptive Question Paper")
    qp.add_page()
    qp.set_auto_page_break(auto=True, margin=15)
    _meta_block(qp, meta)

    W = qp.usable_width
    short_qs = [q for q in questions if q.get("type") == "short"]
    long_qs = [q for q in questions if q.get("type") == "long"]

    if short_qs:
        qp.set_font("Helvetica", "B", 11)
        qp.cell(W, 8, "Section A - Short Answer Questions", new_x="LMARGIN", new_y="NEXT")
        qp.ln(2)
        for q in short_qs:
            qp.set_font("Helvetica", "B", 10)
            qp.multi_cell(W, 7, _safe(f"Q{q['q_no']}.  [{q['marks']} marks]  {q['question']}"))
            qp.ln(10)

    if long_qs:
        qp.ln(2)
        qp.set_font("Helvetica", "B", 11)
        qp.cell(W, 8, "Section B - Long Answer Questions", new_x="LMARGIN", new_y="NEXT")
        qp.ln(2)
        for q in long_qs:
            qp.set_font("Helvetica", "B", 10)
            qp.multi_cell(W, 7, _safe(f"Q{q['q_no']}.  [{q['marks']} marks]  {q['question']}"))
            qp.ln(16)

    # -- Answer Key --
    ak = _PDF("EduCare - Descriptive Answer Key")
    ak.add_page()
    ak.set_auto_page_break(auto=True, margin=15)
    _meta_block(ak, meta)

    W = ak.usable_width

    for q in questions:
        label = "Short" if q.get("type") == "short" else "Long"
        ak.set_font("Helvetica", "B", 10)
        ak.set_x(ak.l_margin)
        ak.multi_cell(W, 7, _safe(f"Q{q['q_no']}. [{label} - {q['marks']} marks]  {q['question']}"))
        ak.set_font("Helvetica", "I", 9)
        ak.set_fill_color(230, 240, 255)
        ak.set_x(ak.l_margin)
        ak.multi_cell(W, 6, _safe(f"  Answer Hint: {q.get('answer_hint', '')}"), fill=True)
        ak.ln(5)

    return bytes(qp.output()), bytes(ak.output())
