#!/bin/bash
gs -q -sPAPERSIZE=a3 -dNOPAUSE -dBATCH -sDEVICE=pdfwrite -sOutputFile=apresentacao.pdf slides/slide_*
