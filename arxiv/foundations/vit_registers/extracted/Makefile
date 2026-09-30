filename=0_main

pdf:
	pdflatex ${filename}
	bibtex ${filename}
	pdflatex ${filename}
	pdflatex ${filename}

supp:
	pdflatex supp
	bibtex supp
	pdflatex supp
	pdflatex supp


clean:
	rm -f ${filename}.{ps,pdf,log,aux,out,dvi,brf,bbl,blg} supp.{ps,pdf,log,aux,out,dvi,brf,bbl,blg}

