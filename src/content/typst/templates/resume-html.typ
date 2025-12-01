#import "@preview/fontawesome:0.6.0": fa-icon
#import "./resume-common.typ": contact, make-entries

#let fontawesome(icon) = {
  html.elem(
    "span",
    attrs: (style: "font-family: 'Font Awesome 7 Free', 'Font Awesome 7 Brands'; font-weight: 400;"),
    fa-icon(icon),
  )
}

#let style(name: "Fr4nk1in", contact-infos: (:), location: none, body) = {
  show underline: it => {
    html.elem("u", it)
  }

  show smallcaps: it => {
    html.elem("span", attrs: (style: "font-variant: small-caps;"), it)
  }

  show heading.where(level: 2): h2 => {
    html.elem("h3", attrs: (style: "border-bottom: 1px solid; font-variant: small-caps;"), h2.body)
  }

  let contact-contents = contact(infos: contact-infos, fa-icon: fontawesome)

  html.elem(
    "blockquote",
    html.elem("p")[A PDF version of this résumé is available #link("/resume.pdf")[here].],
  )

  if location != none {
    text(location)
    linebreak()
  }

  contact-contents.join([ #sym.diamond.stroked.medium ])

  body
}

#let entries = make-entries(
  (left, right) => html.elem("div", attrs: (style: "display: flex; justify-content: space-between;"), {
    html.elem("div", attrs: (style: "text-align: left;"), left)
    html.elem("div", attrs: (style: "text-align: right; white-space: nowrap;"), right)
  }),
  (upper-left, upper-right, lower-left, lower-right) => html.elem(
    "div",
    attrs: (style: "display: flex; flex-direction: column;"),
    {
      html.elem("div", attrs: (style: "display: flex; justify-content: space-between;"), {
        html.elem("div", attrs: (style: "text-align: left;"), upper-left)
        html.elem("div", attrs: (style: "text-align: right; white-space: nowrap;"), upper-right)
      })
      html.elem("div", attrs: (style: "display: flex; justify-content: space-between;"), {
        html.elem("div", attrs: (style: "text-align: left;"), lower-left)
        html.elem("div", attrs: (style: "text-align: right; white-space: nowrap;"), lower-right)
      })
    },
  ),
)
