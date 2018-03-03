package datasets

import (
	"encoding/csv"
	"encoding/json"
	"io/ioutil"
	"os"
	"strconv"

	"gonum.org/v1/gonum/mat"
)

// IrisData structure returned by LoadIris
type IrisData struct {
	Data         [][]float64 `json:"data,omitempty"`
	Target       []float64   `json:"target,omitempty"`
	TargetNames  []string    `json:"target_names,omitempty"`
	DESCR        string      `json:"DESCR,omitempty"`
	FeatureNames []string    `json:"feature_names,omitempty"`
}

// LoadIris load the iris dataset
func LoadIris() (ds *IrisData) {
	filepath := os.Getenv("GOPATH") + "/src/github.com/pa-m/sklearn/datasets/data/iris.json"
	dat, err := ioutil.ReadFile(filepath)
	chk(err)
	ds = &IrisData{}
	err = json.Unmarshal(dat, &ds)
	chk(err)
	return
}

// IrisGetMatrices returns X,Y matrices for iris dataset
func IrisGetMatrices(ds *IrisData) (X, Y *mat.Dense) {
	nSamples, nFeatures, nOutputs := len(ds.Data), len(ds.FeatureNames), 1
	X = mat.NewDense(nSamples, nFeatures, nil)
	X.Apply(func(i, j int, _ float64) float64 {
		return ds.Data[i][j]
	}, X)
	Y = mat.NewDense(nSamples, nOutputs, nil)
	Y.Apply(func(i, _ int, _ float64) float64 {
		return ds.Target[i]
	}, Y)
	return
}

// LoadExamScore loads data from ex2data1 from Andrew Ng machine learning course
func LoadExamScore() (X, Y *mat.Dense) {
	return loadCsv(os.Getenv("GOPATH")+"/src/github.com/pa-m/sklearn/datasets/data/ex2data1.txt", nil, 1)

}

// LoadMicroChipTest loads data from ex2data2 from  Andrew Ng machine learning course
func LoadMicroChipTest() (X, Y *mat.Dense) {
	return loadCsv(os.Getenv("GOPATH")+"/src/github.com/pa-m/sklearn/datasets/data/ex2data2.txt", nil, 1)
}

func loadCsv(filepath string, setupReader func(*csv.Reader), nOutputs int) (X, Y *mat.Dense) {
	f, err := os.Open(filepath)
	chk(err)
	defer f.Close()
	r := csv.NewReader(f)
	if setupReader != nil {
		setupReader(r)
	}
	cells, err := r.ReadAll()
	chk(err)
	nSamples, nFeatures := len(cells), len(cells[0])-nOutputs
	X = mat.NewDense(nSamples, nFeatures, nil)
	X.Apply(func(i, j int, _ float64) float64 { x, err := strconv.ParseFloat(cells[i][j], 64); chk(err); return x }, X)
	Y = mat.NewDense(nSamples, nOutputs, nil)
	Y.Apply(func(i, o int, _ float64) float64 {
		y, err := strconv.ParseFloat(cells[i][nFeatures], 64)
		chk(err)
		return y
	}, Y)
	return
}

func chk(err error) {
	if err != nil {
		panic(err)
	}
}