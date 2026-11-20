#!/usr/bin/env python3
"""Write a plain-text copy (.txt) next to every .html file under a directory,
so the documentation mirrors can be searched with grep.

usage: html2txt.py DIR
"""
import html.parser
import os
import sys

SKIP = {"script", "style", "nav", "header", "footer", "svg", "noscript", "form"}
BLOCK = {"p", "div", "section", "article", "br", "li", "tr", "table", "pre", "h1", "h2", "h3", "h4",
         "h5", "h6", "dt", "dd", "ul", "ol", "blockquote"}


class Text(html.parser.HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out = []
        self.skip = 0
        self.pre = 0

    def handle_starttag(self, tag, attrs):
        if tag in SKIP:
            self.skip += 1
        elif tag == "pre":
            self.pre += 1
            self.out.append("\n")
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self.out.append("\n\n" + "#" * int(tag[1]) + " ")
        elif tag in ("td", "th"):
            self.out.append(" | ")
        elif tag in BLOCK:
            self.out.append("\n")

    def handle_endtag(self, tag):
        if tag in SKIP:
            self.skip = max(0, self.skip - 1)
        elif tag == "pre":
            self.pre = max(0, self.pre - 1)
            self.out.append("\n")
        elif tag in BLOCK:
            self.out.append("\n")

    def handle_data(self, data):
        if self.skip:
            return
        self.out.append(data if self.pre else " ".join(data.split()) + (" " if data[-1:].isspace() else ""))


def convert(path):
    p = Text()
    with open(path, encoding="utf-8", errors="replace") as f:
        p.feed(f.read())
    lines = [line.rstrip() for line in "".join(p.out).split("\n")]
    text, blank = [], 0
    for line in lines:
        blank = blank + 1 if not line.strip() else 0
        if blank <= 1:
            text.append(line)
    with open(os.path.splitext(path)[0] + ".txt", "w", encoding="utf-8") as f:
        f.write("\n".join(text).strip() + "\n")


def main():
    n = 0
    for root, _dirs, files in os.walk(sys.argv[1]):
        for name in files:
            if name.endswith((".html", ".htm")):
                convert(os.path.join(root, name))
                n += 1
    print(f"html2txt: {n} files")


if __name__ == "__main__":
    main()
