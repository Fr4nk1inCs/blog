#let contact-metadata = (
  telephone: (icon: "phone", url-prefix: "tel:"),
  email: (icon: "envelope", url-prefix: "mailto:"),
  linkedin: (icon: "linkedin", url-prefix: "https://www.linkedin.com/in/"),
  github: (icon: "github", url-prefix: "https://github.com/"),
  website: (icon: "globe", url-prefix: "https://"),
)

#let contact(
  infos: (:),
  fa-icon: function,
) = (
  contact-metadata
    .pairs()
    .map(item => {
      let key = item.at(0)
      let meta = item.at(1)
      let value = infos.at(key, default: none)

      if value == none {
        return none
      }

      let icon = fa-icon(meta.icon)
      let url = meta.url-prefix + value

      return icon + " " + link(url, value)
    })
    .filter(item => item != none)
)

#let period(begin, end) = {
  begin + sym.dash.em + end
}

#let make-entries(split, two-row-split) = (
  edu: (
    university: "",
    location: "",
    degree: "",
    begin: "",
    end: "",
  ) => {
    two-row-split(strong(university), location, degree, emph(period(begin, end)))
  },
  work: (
    position: "",
    company: "",
    location: "",
    begin: "",
    end: "",
  ) => {
    two-row-split(strong(position), location, company, emph(period(begin, end)))
  },
  project: (
    title: "",
    role: "",
    location: "",
    begin: "",
    end: "",
  ) => {
    two-row-split(strong(title), location, role, emph(period(begin, end)))
  },
  annotated: (annotation, body) => split(body, emph(annotation)),
)

