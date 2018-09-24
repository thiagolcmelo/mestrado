#!/bin/bash

rm onda_plana_parametros_bons.webm
ffmpeg -r 2 -i saidas/onda_plana_parametros_bons_%02d.png onda_plana_parametros_bons.webm

rm onda_plana_parametros_ruins.webm
ffmpeg -r 2 -i saidas/onda_plana_parametros_ruins_%02d.png onda_plana_parametros_ruins.webm

rm onda_plana_parametros_ruins_stdev.webm
ffmpeg -r 2 -i saidas/onda_plana_parametros_ruins_stdev_%02d.png onda_plana_parametros_ruins_stdev.webm

rm onda_plana_parametros_ruins_skew.webm
ffmpeg -r 2 -i saidas/onda_plana_parametros_ruins_skew_%02d.png onda_plana_parametros_ruins_skew.webm

rm oscilador_harmonico_evoluindo.webm
ffmpeg -r 6 -i saidas/oscilador_harmonico_evoluindo_%03d.png oscilador_harmonico_evoluindo.webm

rm segregacao.webm
rm segregacao_2.webm
cd saidas/
for f in $(ls segregacao_*.png); do convert -resize 2623x1374 -size 2623x1374 $f xc:white +swap -compose over -composite "$(echo $f | cut -d"." -f1).jpg"; done
cd ..
ffmpeg -r 1 -i saidas/segregacao_%02d.jpg segregacao.webm
ffmpeg -i segregacao.webm -vf "pad=width=2663:height=1414:x=20:y=20:color=white" segregacao_2.webm

rm tensao_ingaas.webm
ffmpeg -r 1 -i saidas/tensao_ingaas_%02d.png tensao_ingaas.webm

rm poco_dupla_barreira_potencial_osc.webm
ffmpeg -r 24 -i saidas/poco_dupla_barreira_potencial_osc_%04d.png -s:v 888x500 poco_dupla_barreira_potencial_osc.webm

rm poco_dupla_barreira_evolucao.webm
ffmpeg -r 48 -i saidas/poco_dupla_barreira_evolucao_%04d.png -s:v 888x482 poco_dupla_barreira_evolucao.webm
