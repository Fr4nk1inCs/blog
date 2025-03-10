+++
title = "Template"
description = "A template post that presents the features of this blog."
date = 2025-03-10

[taxonomies]
tags = ["template"]
+++

This is a template post that presents the features of this blog.

## Features

### Typst

Math equations are rendered using [typst](https://typst.app/).

```md
This is an example of an inline math equation: $E = m c^2$, $alpha = 1 / 2$.
```

> This is an example of an inline math equation: $E = m c^2$, $alpha = 1 / 2$.

```md
This is a example of a block math equation:

$$
E = m c^2
$$
```

> This is an example of a block math equation:
> 
> $$
> E = m c^2
> $$

You can also render typst documents:

~~~md
```typ,include=images/test.typ
```
~~~

```typ,include=images/test.typ
```

### Code

```rust
fn main() {
    println!("Hello, World!");
}
```

```rust,copy
fn main() {
    println!("Hello, World!");
}
```

```rust,linenos
fn main() {
    println!("Hello, World!");
}
```

```rust,linenos,linenostart=10
fn main() {
    println!("Hello, World!");
}
```

```rust,linenos,hl_lines=1 5-6
fn main() {
    println!("Hello, World!");
}

fn bar() {
    println!("foo");
}
```

```rust,name=hello.rs,copy
fn main() {
    println!("Hello, World!");
}
```

