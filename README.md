# MoAngelo — Project Page

Project page for **MoAngelo: Motion-Aware Neural Surface Reconstruction for Dynamic Scenes** (Mohamed Ebbed, Zorah Lähner — 3DV 2026).

- Live page: https://mohamed-ebbed.github.io/MoAngelo/
- Paper: https://arxiv.org/abs/2509.15892
- Code: https://github.com/mohamed-ebbed/MoAngelo

## Local preview

```bash
python3 -m http.server 8000
# open http://localhost:8000
```

The site is plain HTML/CSS served by GitHub Pages from the `gh-pages` branch of the [MoAngelo](https://github.com/mohamed-ebbed/MoAngelo) repository; there is no build step. The code lives on `main`.

## Layout

- `index.html` — the page
- `static/css/index.css` — page styles (on top of Bulma)
- `static/js/index.js` — BibTeX copy button, scroll-to-top, play video only while visible
- `static/images/` — teaser, method figure, qualitative comparison renders, social preview (1200×630)
- `static/videos/comparisons.mp4` — dynamic comparison video (H.264, faststart)

## Acknowledgments

Built from the [Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template), which was adopted from the [Nerfies](https://nerfies.github.io/) project page.

## Website License

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This website is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.
