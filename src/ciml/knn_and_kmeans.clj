(ns ciml.knn-and-kmeans
  (:require [clojure.data.priority-map :refer :all]
            [incanter.datasets :refer [get-dataset]]
            [incanter.stats :refer [euclidean-distance]]))

(defn knn-predict [k training-data test-sample features label]
  (->>
   (for [training-sample training-data]
     (euclidean-distance (map test-sample features)
                         (map training-sample features)))
   (zipmap training-data)
   (into (priority-map))
   (take k)
   (keys)
   (map label)
   (frequencies)
   (into (priority-map-by >))
   (ffirst)))

(defn random-value-in-range [low high]
  (-> (- high low) (* (rand)) (+ low)))

(defn random-location [samples]
  (let [lows (apply map min samples)
        highs (apply map max samples)]
    (map random-value-in-range lows highs)))

(defn closest [centers point]
  (->> centers
       (map (partial euclidean-distance point))
       (zipmap centers)
       (into (priority-map))
       (ffirst)))

(defn mean [samples]
  (->> samples
       (apply map +)
       (map #(/ % (count samples)))))

(defn k-means
  ([k training-data features]
     (let [samples (for [sample training-data] (map sample features))
           centers (take k (repeatedly #(random-location samples)))]
       (map (partial zipmap features)
            (k-means samples centers))))
  ([samples centers]
     (let [new-centers
           (->> samples
                (group-by (partial closest centers))
                (vals)
                (map mean))]
       (if (= new-centers centers) centers
           (recur samples new-centers)))))
