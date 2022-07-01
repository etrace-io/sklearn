package modelselection

import (
	"fmt"
	"sort"

	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/mat"

	"github.com/etrace-io/sklearn/base"
	"github.com/etrace-io/sklearn/datasets"
	linearModel "github.com/etrace-io/sklearn/linear_model"
	"github.com/etrace-io/sklearn/metrics"
)

func ExampleCrossValidate() {
	// example adapted from https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
	for _, NJobs := range []int{1, 3} {
		randomState := rand.New(base.NewLockedSource(5))
		diabetes := datasets.LoadDiabetes()
		X, y := diabetes.X.Slice(0, 150, 0, diabetes.X.RawMatrix().Cols).(*mat.Dense), diabetes.Y.Slice(0, 150, 0, 1).(*mat.Dense)
		lasso := linearModel.NewLasso()
		scorer := func(Y, Ypred mat.Matrix) float64 {
			e := metrics.R2Score(Y, Ypred, nil, "").At(0, 0)
			return e
		}
		cvresults := CrossValidate(lasso, X, y, nil, scorer, &KFold{NSplits: 3, Shuffle: true, RandomState: randomState}, NJobs)
		sort.Sort(cvresults)
		fmt.Printf("%.8f\n", cvresults.TestScore)
	}
	// Output:
	// [0.29391770 0.25681807 0.24695688]
	// [0.29391770 0.25681807 0.24695688]

}
