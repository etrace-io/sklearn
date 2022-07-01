package optimize

import (
	"fmt"
	"math"
	"testing"

	"gonum.org/v1/gonum/optimize"
)

func ExamplePowell_Run() {

	settings := &optimize.Settings{
		//MajorIterations: 50,
		//FuncEvaluations: 50,
		//Recorder:        optimize.NewPrinter(),
	}
	method := &Powell{}
	res, err := optimize.Minimize(optimize.Problem{
		Func: func(x []float64) float64 { return 1 - math.Exp(1/(1+x[0]*x[0]+x[1]*x[1]))/math.E },
	}, []float64{10, 20}, settings, method)
	if err != nil {
		panic(err)
	}
	fmt.Printf("%s %.5f\n", res.Status, res.X)
	// Output:
	// MethodConverge [-0.00033 -0.00317]
}

func panics(f func()) (panics bool) {
	defer func() {
		if r := recover(); r != nil {
			panics = true
		}
	}()
	f()
	return
}

func TestPowell(t *testing.T) {
	has := optimize.Available{}
	m := &Powell{}
	uses, err := m.Uses(has)
	if err != nil {
		t.Error(err)
	}
	if uses.Grad || uses.Hess {
		t.Fail()
	}
	if !panics(func() { m.Init(0, 0) }) {
		t.Fail()
	}
	if !panics(func() { m.Init(1, -1) }) {
		t.Fail()
	}
}
