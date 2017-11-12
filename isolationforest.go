package isolationforest

import (
	"math"
	"math/rand"
)

// IsolationForest data type of forest
type IsolationForest struct {
	nTrees     int
	sampleSize int
	trees      []*IsolationTree
	rng        rand.Rand
}

var goroutineWatcher chan struct{}

// Init constructor of IsolationForest
func Init(nTrees int, sampleSize int, rng rand.Rand) *IsolationForest {
	return &IsolationForest{nTrees, sampleSize, make([]*IsolationTree, nTrees), rng}
}

// Build IsolationForest
func (forest *IsolationForest) Build(data [][]float64) {
	dataLen := len(data)
	if (dataLen / forest.sampleSize) < forest.nTrees {
		forest.nTrees = dataLen / forest.sampleSize
	}

	// Shuffle the data in place
	for i := 0; i < 10; i++ {
		for o := range data {
			if o != len(data)-1 {
				n := forest.rng.Intn(o + 1)
				data[o], data[n] = data[n], data[o]
			}
		}
	}

	ch := make(chan *IsolationTree)
	goroutineWatcher = make(chan struct{}, 50)

	for i := 0; i < forest.nTrees; i++ {
		treeData := make([][]float64, forest.sampleSize)
		for j := 0; j < forest.sampleSize; j++ {
			treeData = append(treeData, data[i*forest.sampleSize+j])
		}
		goroutineWatcher <- struct{}{}
		go buildIsolationTreeAsync(treeData, 20, forest.rng, ch)
	}

	for i := 0; i < forest.nTrees; i++ {
		forest.trees = append(forest.trees, <-ch)
	}
	goroutineWatcher = nil
}

// Predict the data
func (forest *IsolationForest) Predict(data [][]float64) []float64 {
	predicted := make([]float64, len(data))

	for _, datapoint := range data {
		predicted = append(predicted, forest.PredictSingle(datapoint))
	}

	return predicted
}

// PredictSingle datapoint
func (forest *IsolationForest) PredictSingle(data []float64) float64 {
	heightSum := 0
	heightAverage := float64(0.0)
	score := 0.0
	heightChan := make(chan int)

	goroutineWatcher = make(chan struct{}, 50)

	for _, tree := range forest.trees {
		goroutineWatcher <- struct{}{}
		go getHeightAsync(tree, data, heightChan)
	}

	for i := 0; i < forest.nTrees; i++ {
		heightSum += <-heightChan
	}
	heightAverage = float64(heightSum) / float64(forest.nTrees)

	goroutineWatcher = nil

	score = math.Pow(
		float64(2),
		-1*(heightAverage/(2*(math.Log(float64(forest.sampleSize-1))+0.577215-(float64(forest.sampleSize-1)/(float64(forest.sampleSize)))))))

	return score
}

// IsolationTree data type of tree
type IsolationTree struct {
	splitFeature int
	splitValue   float64
	height       int
	size         int
	left         *IsolationTree
	right        *IsolationTree
}

func (tree *IsolationTree) getHeight(data []float64) int {
	if tree.left == nil && tree.right == nil {
		return tree.height
	}

	if data[tree.splitFeature] < tree.splitValue {
		return tree.left.getHeight(data)
	}
	return tree.right.getHeight(data)
}

func getHeightAsync(tree *IsolationTree, data []float64, ch chan int) {
	ch <- tree.getHeight(data)
	<-goroutineWatcher
}

// BuildIsolationTree build IsolationTree
func buildIsolationTree(
	data [][]float64,
	currentTreeHeight int,
	heightLimit int,
	rng rand.Rand) *IsolationTree {
	tree := &IsolationTree{0, 0, currentTreeHeight, len(data), nil, nil}

	if len(data) <= 1 {
		return tree
	}

	if currentTreeHeight >= heightLimit {
		return tree
	}

	numberOfFeatures := len(data[0])

	tree.splitFeature = rng.Intn(numberOfFeatures)
	minValue, maxValue := getMinMax(data, tree.splitFeature)
	tree.splitValue = rng.Float64()*(maxValue-minValue) + minValue

	leftData, rightData := splitData(data, tree.splitFeature, tree.splitValue)

	tree.left = buildIsolationTree(leftData, currentTreeHeight+1, heightLimit, rng)
	tree.right = buildIsolationTree(rightData, currentTreeHeight+1, heightLimit, rng)

	return tree
}

func buildIsolationTreeAsync(
	data [][]float64,
	heightLimit int,
	rng rand.Rand,
	ch chan<- *IsolationTree) {
	ch <- buildIsolationTree(data, 0, heightLimit, rng)
	<-goroutineWatcher
}

func getMinMax(data [][]float64, feature int) (float64, float64) {
	var min, max float64

	for i, dp := range data {
		if i == 0 {
			min = dp[feature]
			max = dp[feature]
			continue
		}

		if dp[feature] < min {
			min = dp[feature]
			continue
		}

		if dp[feature] > max {
			max = dp[feature]
			continue
		}
	}

	return min, max
}

func splitData(data [][]float64, feature int, value float64) ([][]float64, [][]float64) {
	left := make([][]float64, len(data)/5)
	right := make([][]float64, len(data)/5)
	for _, dr := range data {
		if dr[feature] < value {
			left = append(left, dr)
		} else {
			right = append(right, dr)
		}
	}
	return left, right
}
