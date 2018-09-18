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
