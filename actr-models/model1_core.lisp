;;;  -*- mode: LISP; Syntax: COMMON-LISP;  Base: 10 -*-
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; 
;;; Author      :Cher YANG
;;; Date        :4/14/2021
;;; 
;;; ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; 
;;; Filename    :model1_core.lisp
;;; Version     :v2.0
;;; 
;;; Description :This lisp script only deals with parameter setting. Main doc can 
;;;              be found in model1_body.lisp
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; 
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
(clear-all)
(define-model model1

;;; --------- PARAMETERS ---------
(sgp ;:seed (100 4)               ; Fixed Randomness
     :er t                      ; Enable randomness
     :esc t                     ; Subsymbolic computations
     :v nil                     ; verbose TRUE
     :trace-detail low     
     :ult nil                   ; Utility Learning Trace
     :act t                     ; Activation trace
     ;---------- activation parameters (3) ----------
     :rt -10                     ; Retrieval Threshold
     :lf .1                   ; Decay Rate
     :bll .3                  ; Base-Level-Learning
     ;:blc 1                    ; Base-Level-Constant
     ;:ol nil                   ; Optimal Learning
     :ans .2                  ; Noise
     :act t
     :ncnar nil
     ;---------- production parameters ----------
     :ul nil                ; Utility learning
     :ppm nil               ; Partial matching
     ;:egs 0                 ; Utility noises
     )
)

