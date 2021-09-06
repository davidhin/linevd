from pathlib import Path

import sastvd as svd

html = """<link
  rel="stylesheet"
  href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.2.0/styles/{}.min.css"
/>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.2.0/highlight.min.js"></script>
<script src="https://cdn.jsdelivr.net/gh/TRSasasusu/highlightjs-highlight-lines.js@1.1.6/highlightjs-highlight-lines.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlightjs-line-numbers.js/2.8.0/highlightjs-line-numbers.min.js"></script>
<script>
  hljs.initHighlightingOnLoad();
  hljs.initLineNumbersOnLoad();
  hljs.initHighlightLinesOnLoad([
    [
        {}
    ],
  ]);
</script>

<style>{}</style>

<pre><code class="language-cpp">{}
</code></pre>"""


def hljs(code, preds, vulns=[], style="idea"):
    """Return highlight JS with predicted lines.

    Example
    style = "idea"
    code = '''int main() {
        int a = 1;
        return a;
    }'''
    preds = {1: 0.5, 2: 0.3}
    """
    hl_lines = []
    for k, v in preds.items():
        hl_lines.append(f'{{ start: {k}, end: {k}, color: "rgba(255, 0, 0, {v})" }}')

    vul_lines = []
    for v in vulns:
        vstyle = f'.hljs-ln-numbers[data-line-number="{v}"] {{  font-weight: bold; color: red; }}'
        vul_lines.append(vstyle)

    vul_lines.append(".hljs-ln-numbers { background-color: white; }")

    return html.format(style, ",".join(hl_lines), "\n".join(vul_lines), code)


def linevd_to_html(cfile, preds, vulns=[], style="idea"):
    """Save a HTML representation of the predicted lines.

    Given a path to a c file and prediction scores for each line in a list, return HTML.
    cfile = svddc.BigVulDataset.itempath(sample.id)
    """
    with open(cfile, "r") as f:
        code = f.read()
    ret = hljs(code, preds, vulns)
    savedir = svd.get_dir(svd.outputs_dir() / "visualise_preds")
    with open(f"{savedir / Path(cfile).name}.html", "w") as f:
        f.write(ret)
    return ret
