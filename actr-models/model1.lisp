;;;  -*- mode: LISP; Syntax: COMMON-LISP;  Base: 10 -*-
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; 
;;; Author      :Cher YANG
;;; Date        :4/14/2021
;;; 
;;; 
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; 
;;; Filename    :mdoel1.py
;;; Version     :v1
;;; 
;;; Description :This instance-based learning model simulates gambling task in HCP dataset.
;;; 
;;; Bugs        : 4.16 Fixed RT issue. RT should be same across conditions
;;;                     Motor preparation
;;;             : 4.16 Seperate productions. -imaginal> should be seperate from 
;;;                     +imaginal>
;;;
;;; To do       : 
;;; 
;;; ----- History -----
;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
;;; General Docs:
;;; 
;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
;;; Public API:
;;;
;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
;;; Design Choices:
;;; 
;;; Task description 
;;; The model plays a card guessing game where the number on a mystery card ("?") 
;;; In order to win or lose money. 
;;; The model attends on the screen, when a card appears ("?"), it retrieves 
;;; from memory history about recent guesses and the outcomes associated with
;;; them. 
;;; Then the model makes a new guess, either "MORE" or "LESS". 
;;; The feedback is then provided and the model learn/encode the feedback.
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; 
;;; Psudocode 
;;; 
;;; p attend-prob ()
;;; p read-prob ()
;;; p recall()
;;; p cannot-recall()
;;; p guess-more ()
;;; p guess-less ()
;;; p detect-feedback()
;;; p encode-feedback()
;;; p end-task()
;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(clear-all)
(define-model model1

;;; --------- PARAMETERS ---------
(sgp ;:seed (1 2)               ; Fixed Randomness
     :er t                      ; Enable randomness
     :esc t                     ; Subsymbolic computations
     :v nil                     ; verbose TRUE
     :trace-detail low     
     :ult nil                   ; Utility Learning Trace
     :act t                     ; Activation trace
     ;---------- activation parameters (3) ----------
     :rt -10                     ; Retrieval Threshold
     :lf 0.5                   ; Decay Rate
     :bll 0.5                  ; Base-Level-Learning
     ;:blc 0                    ; Base-Level-Constant
     ;:ol nil                   ; Optimal Learning
     :ans nil                  ; Noise
     :act t
     :ncnar nil
     ;---------- production parameters ----------
     :ul nil                ; Utility learning
     :ppm nil               ; Partial matching
     :egs 0               ; Utility noises
     )
;;; --------- RANDOM SEED ---------
;(sgp :seed (100 4))

;;; --------- CHUNK TYPE ---------
(chunk-type goal state)
(chunk-type trial probe guess feedback)
(chunk-type history probe guess feedback outcome)


;;; --------- DM ---------
(add-dm
 (start isa chunk) (attending-feedback isa chunk)
 (attending-probe isa chunk) (pressing-key isa chunk) (encoding-feedback)
 (testing isa chunk) (read-feedback isa chunk)
 (win isa chunk) (lose isa chunk) (neutral) (M) (L)
 (goal isa goal state start)
 (win-history-M isa history probe ? guess M outcome win feedback "win")
 (win-history-L isa history probe ? guess L outcome win feedback "win")
 (lose-history-M isa history probe ? guess M outcome lose feedback "lose")
 (lose-history-L isa history probe ? guess L outcome lose feedback "lose")
 (neutral-history-M isa history probe ? guess M outcome neutral feedback "neutral")
 (neutral-history-L isa history probe ? guess L outcome neutral feedback "neutral")
 )

;;; --------- PRODUCTIONS ---------

;;; detect prob. wait for rge screen to change before doing anything
(p attend-probe
    =goal>
      isa      goal
      state    start
    =visual-location>
    ?visual>
     state     free
   ==>
    +visual>               
      cmd      move-attention
      screen-pos =visual-location
    =goal>
      state    attending-probe
)

(p read-probe
    =goal>
      isa      goal
      state    attending-probe
    =visual>
      isa      visual-object
      value    =val
    ?imaginal>
      state    free
   ==>
    +imaginal>
      isa      trial
      probe    =val
      guess    nil
      feedback  nil
    +retrieval>
      isa      history
      ;probe    =val
      outcome  win
    =goal>
      state    testing

)

(p recall
    =goal>
      isa      goal
      state    testing
    =retrieval>
      isa     history
      outcome  win
      guess    =g
    =imaginal>
      isa      trial
      guess    nil
      feedback nil  
    ?imaginal>
      state    free
    ; ?manual>
    ;   state    free
    ?visual>
      state    free
   ==>
    ; +manual>
    ;   cmd      press-key
    ;   key      =g
    =goal>
      state    pressing-key
    +visual>
      cmd      clear
    *imaginal>
      guess    =g
)

(p cannot-recall
    =goal>
      isa      goal
      state    testing
    =imaginal>
      isa      trial
      guess    nil
      feedback nil 
    ?imaginal>
      state    free
    ?retrieval>
      buffer   failure
    ; ?manual>
    ;   state    free
    ?visual>
      state    free
   ==>
    ; +manual>
    ;   cmd      press-key
    ;   key      "M"  
    =goal>
      state    pressing-key
    +visual>
     cmd      clear
    *imaginal>
      guess    nil
)

(p guess-more
    =goal>
      isa      goal
      state    pressing-key
    =imaginal>
      isa      trial
      - guess    nil
      feedback nil
    ?imaginal>
      state    free
    ?manual>
      preparation free
      execution free
      processor free
  ==>
    +manual>
      ;cmd      press-key
      ;key      "K"
      cmd punch
      finger index
      hand right
    =goal>
      state    read-feedback
    =imaginal>
    ; +visual>
    ;   cmd      clear
    ; *imaginal>
    ;   guess    "K"
  )

(p guess-less
    =goal>
      isa      goal
      state    pressing-key
    =imaginal>
      isa      trial
      - guess    nil
      feedback nil
    ?imaginal>
      state    free
    ?manual>
      preparation free
      execution free
      processor free
    ?visual>
      state    free
  ==>
    +manual>
      ;cmd      press-key
      ;key      "F"
      cmd punch
      finger index
      hand left
    =goal>
      state    read-feedback
    =imaginal>
    ; +visual>
    ;   cmd      clear
    ; *imaginal>
    ;   guess    "F"
  )

;;; detect feedback. wait for rge screen to change before doing anything
(p detect-feedback
    =goal>
      isa      goal
      state    read-feedback
    =visual-location>
    ?visual>
      state    free
   ==>
    +visual>
      cmd      move-attention
      screen-pos =visual-location
    =goal>
      state    attending-feedback
)

; TODO: revise imaginal buffer
(p encode-feedback
    =goal>
      isa      goal
      state    attending-feedback
    =visual>
      isa      visual-object
      value    =val
    =imaginal>
      isa      trial
      probe     =p
      guess     =g
      feedback nil
    ?visual>
      state    free
    ?imaginal>
      state    free
  ==>
   +imaginal>
      isa      history
      probe     =p
      guess     =g
      outcome   =val
   =goal>
      state    encoding-feedback
   +visual>
      cmd      clear
)

(p end-task
    =goal>
        isa      goal
        state    encoding-feedback
    ?imaginal>
        state    free
  ==>
    -imaginal>
    =goal>
        state    start
  )


(goal-focus goal)
)

