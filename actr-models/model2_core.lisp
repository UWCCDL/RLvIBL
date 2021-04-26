;;;  -*- mode: LISP; Syntax: COMMON-LISP;  Base: 10 -*-
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; 
;;; Author      :Cher YANG
;;; Date        :4/14/2021
;;; 
;;; ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; 
;;; Filename    :model2_core.lisp
;;; Version     :v2.0
;;; 
;;; Description :This lisp script only deals with parameter setting. Main doc can 
;;;              be found in model2_body.lisp
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; 
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
(clear-all)
(define-model model2

;;; --------- PARAMETERS ---------
(sgp ;:seed (200 4)               ; Fixed Randomness
     :er t                      ; Enable randomness
     :esc t                     ; Subsymbolic computations
     :v nil                     ; verbose TRUE
     :trace-detail low     
     :ult nil                   ; Utility Learning Trace
     :act nil                   ; Activation trace
     ;---------- activation parameters ----------
     ;:rt -2                    ; Retrieval Threshold
     ;:lf nil                   ; Decay Rate
     ;:bll nil                  ; Base-Level-Learning
     ;:blc 0                    ; Base-Level-Constant
     ;:ol nil                   ; Optimal Learning
     :ans nil                   ; Noise
     :act nil
     :ncnar nil
     ;---------- production parameters ----------
     :ul t                      ; Utility learning
     :ult nil                     ; Utility learning trace
     :cst nil                     ; Conflict set trace
     ;:ppm nil                   ; Partial matching
     :alpha 0.2                 ; Learning rate
     :egs 0.1                   ; Utility noises
     ;:pca                       ; Production
     )

)
