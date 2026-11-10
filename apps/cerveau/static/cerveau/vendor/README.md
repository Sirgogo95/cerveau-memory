# Vendored libraries

Third-party browser libraries served from here instead of a CDN, so the chat page
loads nothing from the internet. Update by downloading the new version from the same
source and replacing the file; keep the license header at the top.

| File | Library | Version | Source | License | SHA-256 |
|---|---|---|---|---|---|
| `marked.umd.js` | [marked](https://github.com/markedjs/marked): Markdown to HTML | 18.1.1 | `https://cdn.jsdelivr.net/npm/marked@18.1.1/lib/marked.umd.js` | MIT | `b82b9334406d8bdd16b6236c08de1cb65b2a859873ca9503d9cc0b292ca3c12a` |
| `purify.min.js` | [DOMPurify](https://github.com/cure53/DOMPurify): HTML sanitizer | 3.4.17 | `https://cdn.jsdelivr.net/npm/dompurify@3.4.17/dist/purify.min.js` | Apache-2.0 or MPL-2.0 | `cd37898f9782adf54820c413191b5691803f0f137c799d6edcfd38523b4c2c31` |
| `fonts/hanken-grotesk.woff2` | [Hanken Grotesk](https://github.com/marcologous/hanken-grotesk), variable 400 to 700, Latin subset | v12 | `https://fonts.gstatic.com/s/hankengrotesk/v12/ieVn2YZDLWuGJpnzaiwFXS9tYtpd59A.woff2` | OFL 1.1 (`fonts/OFL-HankenGrotesk.txt`) | `e9201eddf1d41d0b62253295d869ce3cf65768f7102b797f02c7f8c876b4a9d5` |
| `fonts/hanken-grotesk-italic.woff2` | Hanken Grotesk italic | v12 | `https://fonts.gstatic.com/s/hankengrotesk/v12/ieVl2YZDLWuGJpnzaiwFXS9tYtpY19K7DQ.woff2` | OFL 1.1 | `bb432642d7e97d11bd8f7adbcb79dc69b772211c9b6eae5251a969496674d299` |
| `fonts/instrument-serif.woff2` | [Instrument Serif](https://github.com/Instrument/instrument-serif), Latin subset | v5 | `https://fonts.gstatic.com/s/instrumentserif/v5/jizBRFtNs2ka5fXjeivQ4LroWlx-6zUTjg.woff2` | OFL 1.1 (`fonts/OFL-InstrumentSerif.txt`) | `5eb09b5ac0e28b67c2f041c8ba6d244604ca0c0980d65912ab2d47fed84ddc31` |
| `fonts/instrument-serif-italic.woff2` | Instrument Serif italic | v5 | `https://fonts.gstatic.com/s/instrumentserif/v5/jizHRFtNs2ka5fXjeivQ4LroWlx-6zAjjH7M.woff2` | OFL 1.1 | `5a51946dfffa82972bc98745359c46761515641fda557c25116459a9f83da4a7` |

The Latin subsets cover English, Spanish, and French (á, é, ñ, ç, ¿, ¡, œ).

Model replies are Markdown; `chat.js` renders them with marked and always passes the
result through DOMPurify before it reaches the page, since a reply can contain HTML
(for example from a stored memory) that must never run.
