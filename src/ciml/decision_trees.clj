(ns ciml.decision-trees
  (:require [clojure.data.priority-map :refer [priority-map-by]]
            [clojure.pprint :refer [pprint]]
            [incanter.core :refer [dataset]]))

(def training-data
  (dataset
   [:rating :easy?  :ai? :sys? :thy? :morning?]
   [[2           1     1     0     1         0]
    [2           1     1     0     1         0]
    [2           0     1     0     0         0]
    [2           0     0     0     1         0]
    [2           0     1     1     0         1]
    [1           1     1     0     0         0]
    [1           1     1     0     1         0]
    [1           0     1     0     1         0]
    [0           0     0     0     0         1]
    [0           1     0     0     1         1]
    [0           0     1     0     1         0]
    [0           1     1     1     1         1]
    [-1          1     1     1     0         1]
    [-1          0     0     1     1         0]
    [-1          0     0     1     0         1]
    [-1          1     0     1     0         1]
    [-2          0     0     1     1         0]
    [-2          0     1     1     0         1]
    [-2          1     0     1     0         0]
    [-2          1     0     1     0         1]]))

(defn label [row]
  (if (< (row :rating) 0) :hate :like))

(defn labels [rows]
  (->> rows (map label)))

(defn most-common-label [rows]
  (->> rows labels frequencies (into (priority-map-by >)) ffirst))

(defn unambiguous [rows]
  (->> rows labels (into #{}) count (= 1)))

(defn split-by-feature-values [rows feature]
  (->> rows (group-by feature) vals))

(defn majority-vote-count [rows]
  (->> rows (group-by label) vals (map count) (apply max)))

(defn score-feature [rows feature]
  (->> (split-by-feature-values rows feature) (map majority-vote-count) (apply +)))

(defn score-features [rows features]
  (zipmap features (map (partial score-feature rows) features)))

(defn feature-set [dataset]
  (let [column-set (->> dataset :column-names (into #{}))]
    (disj column-set :rating)))

(defn relevant-feature? [rows feature]
  (< 1 (count (split-by-feature-values rows feature))))

(defn retain-relevant [rows features]
  (into #{} (filter (partial relevant-feature? rows) features)))

(defn decision-tree-train
  ([dataset] (decision-tree-train (:rows dataset) (feature-set dataset)))
  ([rows features]
     (if (or (unambiguous rows) (empty? features))
       (most-common-label rows)
       (let [relevant-features  (retain-relevant rows features)
             scored-features    (score-features rows relevant-features)
             sorted-by-score    (into (priority-map-by >) scored-features)
             winning-feature    (ffirst sorted-by-score)
             split-rows         (group-by winning-feature rows)
             remaining-features (disj relevant-features winning-feature)]
         {:feature winning-feature
          0 (decision-tree-train (split-rows 0) remaining-features)
          1 (decision-tree-train (split-rows 1) remaining-features)}))))

(defn decision-tree-train-with-limit
  ([dataset limit]
     (decision-tree-train-with-limit (:rows dataset) (feature-set dataset) limit))
  ([rows features limit]
     (if (or (= 0 limit) (unambiguous rows) (empty? features))
       (most-common-label rows)
       (let [relevant-features  (retain-relevant rows features)
             scored-features    (score-features rows relevant-features)
             sorted-by-score    (into (priority-map-by >) scored-features)
             winning-feature    (ffirst sorted-by-score)
             split-rows         (group-by winning-feature rows)
             remaining-features (disj relevant-features winning-feature)]
         {:feature winning-feature
          0 (decision-tree-train-with-limit (split-rows 0) remaining-features (dec limit))
          1 (decision-tree-train-with-limit (split-rows 1) remaining-features (dec limit))}))))

(defn decision-tree-test [tree test-point]
  (if (map? tree)
    (decision-tree-test (tree (test-point (tree :feature))) test-point)
    tree))
