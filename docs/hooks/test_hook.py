def on_post_page(output, page, config):
    path = str(page.file.src_uri)
    if not path.endswith(".ipynb"):
        return output

    output = output.replace(
        """<div class="highlight-ipynb hl-python">""",
        """<div class="language-python highlight">"""
        )

    return output
