"""Minimal PDF byte builder for tests.

A hand-written PDF keeps reportlab out of the test dependencies. The output is a
valid single-page PDF with a selectable text object, which is what pypdf's
extract_text() needs.
"""


def _escape(text: str) -> str:
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def build_pdf_bytes(text: str) -> bytes:
    """Return the bytes of a one-page PDF containing `text`."""
    lines = text.splitlines() or [text]
    content_lines = "BT /F1 12 Tf 72 720 Td 14 TL\n"
    for line in lines:
        content_lines += f"({_escape(line)}) Tj T*\n"
    content_lines += "ET"
    content = content_lines.encode("latin-1", errors="replace")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"

    xref_pos = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n"
        f"{xref_pos}\n%%EOF\n"
    ).encode()
    return bytes(out)