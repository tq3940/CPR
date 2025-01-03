#!/bin/sh -x

sh runBPRMF.sh
sh runCPRMF.sh

sh runLightGCN.sh
sh runLightGCNCPR.sh

sh runComiRec.sh
sh runComiRecCPR.sh