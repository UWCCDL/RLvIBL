;;;  -*- mode: LISP; Syntax: COMMON-LISP;  Base: 10 -*-
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; 
;;; Author      :Cher YANG
;;; Date        :4/14/2021
;;; 
;;; 
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; 
;;; Filename    :mdoel2.lisp
;;; Version     :v1
;;; 
;;; Description :This reinforcement learning model simulates gambling task in HCP dataset.
;;; 
;;; Bugs        : Fix RT issue. RT should be same across conditions
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
;;; The model attends on the screen, when a card appears ("?"), it presses either
;;; "MORE"(K) or "LESS"(J) key, and receives reward/punishment.
;;; The feedback is then provided and the model learn/encode the feedback.
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; 
;;; Psudocode 
;;; 
;;; p attend-prob ()
;;; p guess-more ()
;;; p guess-less ()
;;; p detect-feedback()
;;; p encode-reward()
;;; p encode-punishment()
;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(clear-all)
(define-model model2

;;; --------- PARAMETERS ---------
(sgp :seed (200 4)               ; Fixed Randomness
     :er t                      ; Enable randomness
     :esc t                     ; Subsymbolic computations
     :v t                     ; verbose TRUE
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
     ;:ult t                     ; Utility learning trace
     ;:cst t                     ; Conflict set trace
     ;:ppm nil                   ; Partial matching
     :alpha 0.2                 ; Learning rate
     :egs 0.1                   ; Utility noises
     ;:pca                       ; Production
     )

;;; --------- CHUNK TYPE ---------
(chunk-type goal state)
(chunk-type trial probe guess feedback)
(chunk-type history probe guess feedback outcome)


;;; --------- DM ---------
(add-dm
 (start isa chunk) (attending-feedback isa chunk)
 (attending-probe isa chunk)
 (testing isa chunk) (read-feedback isa chunk) (encoding-feedback isa chunk)
 (win isa chunk) (lose isa chunk) (neutral isa chunk) (M) (L)
 (goal isa goal state start)
 (win-history-M isa history probe ? guess J outcome win feedback "win")
 (win-history-L isa history probe ? guess K outcome win feedback "win")
 (lose-history-M isa history probe ? guess J outcome lose feedback "lose")
 (lose-history-L isa history probe ? guess K outcome lose feedback "lose")
 (neutral-history-M isa history probe ? guess J outcome neutral feedback "neutral")
 (neutral-history-L isa history probe ? guess K outcome neutral feedback "neutral")
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

(p press-more
    =goal>
      isa      goal
      state    testing
    =imaginal>
      isa      trial
      guess    nil
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
      ;key      "K"
      cmd punch
      finger index
      hand right
    =goal>
      state    read-feedback  
    +visual>
      cmd      clear
    *imaginal>
      guess    "K"
  )

(p press-less
    =goal>
      isa      goal
      state    testing
    =imaginal>
      isa      trial
      guess    nil
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
    +visual>
      cmd      clear
    *imaginal>
      guess    "F"
  )

;;; detect feedback. wait for rge screen to change before doing anything
(p detect-feedback
    =goal>
      isa      goal
      state    read-feedback
    =visual-location>
    ?visual>
      state    free
    ?manual>
      state free
   ==>
    +manual>
      cmd clear
    +visual>
      cmd      move-attention
      screen-pos =visual-location
    =goal>
      state    attending-feedback
)

(p encode-reward
    =goal>
      isa      goal
      state    attending-feedback
    =visual>
      isa      visual-object
      value    "win"
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
      outcome   "win"
   +visual>
        cmd     clear
   =goal>
      state    encoding-feedback
)

(p encode-punishment
    =goal>
      isa      goal
      state    attending-feedback
    =visual>
      isa      visual-object
      value    "lose"
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
      outcome   "lose"
   +visual>
        cmd     clear
   =goal>
      state    encoding-feedback
)

(p encode-neutral
    =goal>
      isa      goal
      state    attending-feedback
    =visual>
      isa      visual-object
      value    "neutral"
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
      outcome   "neutral"
   +visual>
        cmd     clear
   =goal>
      state    encoding-feedback
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

;------------ reward ------------
(spp encode-reward :reward 1)
(spp encode-neutral  :reward 0)
(spp encode-punishment :reward -1)
)

