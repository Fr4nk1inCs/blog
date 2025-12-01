
#import "templates/mod.typ": sys-is-html-target
#import "templates/resume-pdf.typ": entries as pdf-entries, style as pdf-style
#import "templates/resume-html.typ": entries as html-entries, style as html-style

#let (style, edu-entry, work-entry, project-entry, annotated-entry) = if sys-is-html-target {
  (html-style, html-entries.edu, html-entries.work, html-entries.project, html-entries.annotated)
} else {
  (
    pdf-style.with(accent: rgb("#26428b"), font: ("Libertinus Serif", "SongTi SC")),
    pdf-entries.edu,
    pdf-entries.work,
    pdf-entries.project,
    pdf-entries.annotated,
  )
}

#show: style.with(
  name: "Shen Fu",
  location: [No. 100, Fuxing Rd., High-Tech District, Hefei, Anhui Province, China 230031],
  contact-infos: (
    email: "sh.fu@outlook.com",
    github: "Fr4nk1inCs",
  ),
)

== Research Interests

LLM inference optimization, System for MoE.

== Research Projects

#project-entry(
  title: "Parallelism Planning for MoE Inference with Dynamic Top-K Routing",
  role: "Core Member",
  location: "ADSL, USTC",
  begin: "Mar 2025",
  end: "Aug 2025",
)
- An inference framework for dymamic top-$k$ routing MoE models, which automatically plans parallelism strategies to maximize throughput on prefill-dominated workloads.
- Paricipated in the implementation of the model profiler, adoption of dynamic top-$k$ routing, pipeline parallelism enhancements, and the design of the parallelism planner.

== Publications

Jin, Z., #underline[*Fu, S.*], Tang, C., Bai, Y., Wang, S., Zhu, J., Fang, C., Gong, P., & Li, C. (2025). SMIDT: High-Performance Inference Framework for MoE Models with Dynamic Top-K Routing. (Under review)

== Education

#edu-entry(
  university: "University of Science and Technology of China",
  location: "Hefei, Anhui",
  degree: "B.E. in Computer Science and Technology",
  begin: "Sep 2020",
  end: "Jun 2024",
)
- #link("https://sgy.ustc.edu.cn")[School of the Gifted Young]
- GPA: 3.92/4.30, Rank: top 8%

#edu-entry(
  university: "University of Science and Technology of China",
  location: "Hefei, Anhui",
  degree: "M.E. in Computer Science and Technology",
  begin: "Sep 2024",
  end: "Present",
)
- Advisor: Prof. #link("https://cs.ustc.edu.cn/2020/0828/c23239a615416/pagem.htm")[Cheng Li]
- GPA: 4.13/4.30

== Honors & Scholarships

- #annotated-entry("Oct 2023, USTC")[Qiangwei "Yuanzhi" Scholarship (*Top 3%*)]
- #annotated-entry("Jan 2023, USTC")[Jianghuai & NIO Automobile Scholarship]
- #annotated-entry("Jan 2022, USTC")[Cheng Linyi Scholarship]
- #annotated-entry("Sep 2021, USTC")[Outstanding Freshman Scholarship, Grade 2]

== Miscellaneous

#strong(smallcaps[Service])
- USENIX ATC #(sym.quote.single.r)25 Artifact Evaluation Committee

#strong(smallcaps[Teaching])
- #annotated-entry(
    "2023 Autumn, USTC",
  )[T.A. for _Compiler Principles and Techniques_ (Instructor: Prof. Cheng Li)]


#strong(smallcaps[Open Source Contributions])
- [#link("https://github.com/sgl-project/sglang")[sgl-project/sglang]] #link("https://github.com/sgl-project/sglang/pull/6121")[feat: add dp attention support for Qwen 2/3 MoE models (\#6121)]

#strong(smallcaps[Skills])
- *Languages*: Mandarin Chinese (Native), English (Fluent)
- *Programming*: Python, C/C++, Lua, Shell Script
- *Frameworks*: PyTorch, vLLM, SGLang
