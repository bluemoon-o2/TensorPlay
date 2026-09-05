# 白皮书 LaTeX 编译（本机备忘）

TeX Live 安装在 Windows 盘，WSL 内直接调用 exe：

- `pdflatex`: `/mnt/e/texlive/2025/bin/windows/pdflatex.exe`
- `latexmk`: `/mnt/e/texlive/2025/bin/windows/latexmk.exe`
- Windows 侧对应 `E:\texlive\2025\bin\windows\`

exe 只认 Windows 路径，WSL 的 `/home` 对它不可见。编译白皮书时把
`docs/whitepaper/` 的 `main.tex`、`glossary.tex`、`sections/*.tex` 同步到
`/mnt/e/tmp_tp_whitepaper/`，在该目录执行：

```
latexmk.exe -pdf -interaction=nonstopmode -halt-on-error main.tex
```

完成后把 `main.pdf` 拷回 `docs/whitepaper/`。

本仓库 LaTeX 常见编译错误：`\path{}` 里不能放数学（`$>$`、`$\rightarrow$`）；
`\texttt{}` 里的下划线必须写成 `\_`；caption 里不要内嵌 sed 的 `\1` 反向引用。
