## Instructions for PRs

If you merge a PR or push a change, please make sure the paper compiles.

## Development
- Download MacTeX
https://mirror.ctan.org/systems/mac/mactex/MacTeX.pkg

UI
- Use VisualStudio's Latex Workshop for development - https://marketplace.visualstudio.com/items?itemName=James-Yu.latex-workshop
- There's a `Makefile` if you don't want to use Visual Studio.

- Latex with vscode: https://docs.google.com/document/d/1XmVAzqqdt0FGOrWDHe3ilaJwLq1bueQntCxo5zW1ENg/edit#heading=h.1xfaqv2vzxs0 

## Text

If we have one line per sentence, it makes tracking changes + merging easier.


## Figures

- Figures should follow guidelines for color scheme using https://docs.google.com/presentation/d/1EhlV5nE0pj--YGkB62zXG1Dvk1dDuz-47e8dPswM1l8/edit#slide=id.g2087f16ab6e_0_1936
- Colors (hexcodes) can be found in `https://github.com/facebookresearch/moviegen_paper/blob/main/plot_scripts/moviegen_plot_template.py#L264`
- Please check in the source (pptx/keynote/graffle) files of your figures
- Please export figures as **PDF** before adding to the paper. No PNG, JPEG or other formats.

## Plots

Run 
```
pip3 install matplotlib
pip3 install seaborn
```

### Scripts

These are based off the `llama3_paper` repo.

`plot_scripts/moviegen_plot_templates.py` has the main plot template for the project
See `plot_scripts/make_moviegen_t2v_plots.py` for an example
