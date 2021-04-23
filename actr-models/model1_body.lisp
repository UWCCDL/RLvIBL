;;;  -*- mode: LISP; Syntax: COMMON-LISP;  Base: 10 -*-
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; 
;;; Author      :Cher YANG
;;; Date        :4/14/2021
;;; 
;;; 
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; 
;;; Filename    :model1_body.py
;;; Version     :v2.1
;;; 
;;; Description :This declarative model simulates gambling task in HCP dataset.
;;; 
;;; Bugs        : 4.16 1) Fixed RT issue. RT should be same across conditions
;;;                     Motor preparation
;;;             :      2) Seperated productions. -imaginal> should be seperate from
;;;                     +imaginal>
;;;             :      3) Added consistent productions as model2
;;;                     guess-more and guess-less
;;;             : 4.18 1) Seperated core.lisp and body.lisp code of model1.
;;;                    The parameter setting is now working
;;;             : 4.19 1) Fixed parameter setting
;;;             :      2) added evaluate-more and evaluate less
;;;             :      3) fixed model1 learning bug. model1 does learn, but less flexible
;;;             :      4)

;;;
;;;
;;; To do       : TODO: null RT issue -> neutral trials
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

;;; --------- RANDOM SEED ---------
;(sgp :seed (100 4))

;;; --------- CHUNK TYPE ---------
(chunk-type goal state)
; (chunk-type trial probe guess feedback)
(chunk-type history probe guess feedback)


;;; --------- DM ---------
(add-dm
 (start isa chunk) (attending-feedback isa chunk)
 (attending-probe isa chunk) (pressing-key isa chunk) 
 (encoding-feedback isa chunk) (read-feedback isa chunk)
 (evaluating-history isa chunk)(recalling-history isa chunk) 
 (win isa chunk) (lose isa chunk) (neutral) (M) (L)
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
      state    evaluating-history

)


;;; Eval history

(p evaluate-more
  =goal>
      isa      goal
      state    evaluating-history
  ?imaginal>
      state    free
  ?visual>
      state    free
  =imaginal>
      isa     history
      - probe   nil
==>
  =goal>
      state    recalling-history
  +retrieval>
      isa      history
      guess    M
  =imaginal>
  )

(p evaluate-less
  =goal>
      isa      goal
      state    evaluating-history
  ?imaginal>
      state    free
  =imaginal>
      isa     history
      - probe   nil
==>
  =goal>
      state    recalling-history
  +retrieval>
      isa      history
      guess    L
  =imaginal>
  )

(p recall-win
    =goal>
      isa      goal
      state    recalling-history
    =retrieval>
      isa       history
      feedback  "Reward"
      guess     =g
    =imaginal>
      isa      history
      - probe  nil
    ?imaginal>
      state    free
   ==>
    ; +manual>
    ;   cmd      press-key
    ;   key      =g
    =goal>
      state    pressing-key
    *imaginal>
      guess    =g
)

(p recall-lose-M
    =goal>
      isa      goal
      state    recalling-history
    =retrieval>
      isa       history
      - feedback  "Reward"  ; include neutral memory
      guess     M
    =imaginal>
      isa      history
      - probe  nil
    ?imaginal>
      state    free
   ==>
    =goal>
      state    pressing-key
    *imaginal>
      guess    L ; reverse
)

(p recall-lose-L
    =goal>
      isa      goal
      state    recalling-history
    =retrieval>
      isa       history
      - feedback  "Reward"
      guess     L
    =imaginal>
      isa      history
      - probe  nil
    ?imaginal>
      state    free
   ==>
    =goal>
      state    pressing-key
    *imaginal>
      guess    M ; reverse
)

(p cannot-recall
    =goal>
      isa      goal
      state    recalling-history
    ?imaginal>
      state    free
    ?retrieval>
      buffer   failure
   ==>
    =goal>
      state    pressing-key
    ; *imaginal>
    ;   guess    nil 
)

(p guess-more
    =goal>
      isa      goal
      state    pressing-key
    =imaginal>
      isa      history
      - probe  nil
      - guess    L
      ;feedback nil
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
  )

(p guess-less
    =goal>
      isa      goal
      state    pressing-key
    =imaginal>
      isa       history
      - probe   nil
      - guess   M   ; this allows nil to press less
      ;feedback nil
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
  )

;;; detect feedback. wait for screen to change before doing anything

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
      state  attending-feedback
    =visual>
      isa      visual-object
    - value "?" 
      value    =val
    =imaginal>
      isa      history
      - probe     nil
      - guess     nil
      feedback nil
    ?visual>
      state    free
    ?imaginal>
      state    free
    ?manual>
      state free
    
  ==>
   *imaginal>
      feedback   =val
   =visual>
   =goal>
      state    encoding-feedback
 )

(p end-task
    =goal>
        isa      goal
        state    encoding-feedback
    ?imaginal>
        state    free
        buffer full
  ==>
    -imaginal>
    =goal>
        state    start
  )


(goal-focus goal)





