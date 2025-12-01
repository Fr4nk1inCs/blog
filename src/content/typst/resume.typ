#import "templates/mod.typ": sys-is-svg-in-html
#import "templates/resume-pdf.typ": entries, style

#let edu-entry = entries.edu
#let work-entry = entries.work
#let project-entry = entries.project
#let annotated-entry = entries.annotated

#set text(fill: rgb(sys.inputs.at("main-color", default: "#000000")))

#show: style.with(
  name: "Shen Fu",
  location: [No. 100, Fuxing Rd., High-Tech District, Hefei, Anhui Province, China 230031],
  contact-infos: (
    email: "sh.fu@outlook.com",
    github: "Fr4nk1inCs",
  ),
  paper: "us-letter",
  margin: if sys-is-svg-in-html { (x: 0pt, y: 20pt) } else { 0.75in },
  extra-page-settings: if sys-is-svg-in-html {
    (height: auto)
  } else {
    ()
  },
  accent: rgb(sys.inputs.at("accent-color", default: "#26428b")),
  font-size: if sys-is-svg-in-html { 16pt } else { 12pt },
  font: "Libertinus Serif",
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

#[
  #show "Shen Fu": strong
  #show "Shen Fu": underline
  #bibliography(
    "reference.bib",
    style: "association-for-computing-machinery",
    full: true,
    title: none,
  )
]

== Education

#edu-entry(
  university: "University of Science and Technology of China",
  location: "Hefei, Anhui",
  degree: "M.E. in Computer Science and Technology",
  begin: "Sep 2024",
  end: "Present",
)
- Advisor: Prof. #link("https://cs.ustc.edu.cn/2020/0828/c23239a615416/pagem.htm")[Cheng Li]
- GPA: 4.13/4.30

#edu-entry(
  university: "University of Science and Technology of China",
  location: "Hefei, Anhui",
  degree: "B.E. in Computer Science and Technology",
  begin: "Sep 2020",
  end: "Jun 2024",
)
- #link("https://sgy.ustc.edu.cn")[School of the Gifted Young]
- GPA: 3.92/4.30, Rank: top 8%

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
