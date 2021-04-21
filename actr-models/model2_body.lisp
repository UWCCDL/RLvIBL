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
;;; Version     :v2.0
;;; 
;;; Description :This reinforcement learning model simulates gambling task in HCP dataset.
;;; 
;;; Bugs        : 4.16 Fixed RT issue. RT should be same across conditions
;;;               Motor preparation
;;;               4.18 Seperate core and body
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
;;; p read-prob ()
;;; p guess-more ()
;;; p guess-less ()
;;; p detect-feedback()
;;; p encode-reward()
;;; p encode-punishment()
;;; p encode-neutral()
;;; p end-task()
;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;; --------- CHUNK TYPE ---------
(chunk-type goal state)
(chunk-type history probe guess feedback)


;;; --------- DM ---------
(add-dm
 (start isa chunk) (attending-feedback isa chunk)
 (attending-probe isa chunk) (pressing-key isa chunk) 
 (encoding-feedback isa chunk) (read-feedback isa chunk)
 (neutral isa chunk) (M isa chunk) (L isa chunk)
 (goal isa goal state start)
 (win-history-M isa history probe "?" guess M feedback "Reward")
 (win-history-L isa history probe "?" guess L feedback "Reward")
 (lose-history-M isa history probe "?" guess M feedback "Punishment")
 (lose-history-L isa history probe "?" guess L feedback "Punishment")
 (neutral-history-M isa history probe "?" guess M feedback "Neutral")
 (neutral-history-L isa history probe "?" guess L feedback "Neutral")
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
      isa      history
      probe    =val
    +retrieval>
      isa      history
      probe    =val
      feedback  "Reward"
    =goal>
      state    pressing-key

)

(p guess-more
    =goal>
      isa      goal
      state    pressing-key
    =imaginal>
      isa      history
      guess    nil
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
      guess    M
  )

(p guess-less
    =goal>
      isa      goal
      state    pressing-key
    =imaginal>
      isa      history
      guess    nil
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
      guess    L
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

(p encode-feedback
    =goal>
      isa      goal
      state    attending-feedback
    =visual>
      isa      visual-object
      value    =val
    =imaginal>
      isa      history
      probe     =p
      guess     =g
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