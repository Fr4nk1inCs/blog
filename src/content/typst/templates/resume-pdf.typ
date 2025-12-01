#import "@preview/fontawesome:0.6.0": fa-icon
#import "./resume-common.typ": contact, make-entries

#let style(
  name: "Fr4nk1in",
  font: ("Libertinus Serif",),
  font-size: 12pt,
  title-size: 24pt,
  paper: "us-letter",
  accent: rgb(0, 0, 0),
  lang: "en",
  contact-infos: (:),
  location: none,
  body,
) = {
  set document(author: name, title: "Résumé of " + name)
  set text(
    font: font,
    size: font-size,
    lang: lang,
    ligatures: false,
  )
  set page(paper: paper, margin: 0.75in)
  set par(linebreaks: "optimized", justify: true)

  show heading: set text(fill: accent)

  show heading.where(level: 1): set text(size: title-size)
  show heading.where(level: 1): set align(center)
  show heading.where(level: 2): it => {
    pad(top: 0pt, bottom: -font-size, [#smallcaps(it.body)])
    line(length: 100%, stroke: 1pt)
  }

  let contact-contents = contact(infos: contact-infos, fa-icon: fa-icon)

  align(center, {
    heading(level: 1, name)
    if location != none {
      text(location)
      linebreak()
    }
    text(style: "italic", contact-contents.join(h(1em)))
  })

  body
}

#let entries = make-entries(
  (left, right) => (left + h(1fr) + right),
  (upper-left, upper-right, lower-left, lower-right) => (
    upper-left + h(1fr) + upper-right + linebreak() + lower-left + h(1fr) + lower-right
  ),
)

