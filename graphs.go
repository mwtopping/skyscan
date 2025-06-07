package main

import (
	//	"bytes"
	"fmt"
	//	"html/template"
	//	"io"
	//	"log"
	"math/rand"
	"net/http"

	"github.com/go-echarts/go-echarts/v2/charts"
	"github.com/go-echarts/go-echarts/v2/opts"
	// chartrender "github.com/go-echarts/go-echarts/v2/render"
)

//type Renderer interface {
//	Render(w io.Writer) error
//}
//
//type snippetRenderer struct {
//	c       interface{}
//	before  []func()
//	baseTpl string
//}
//
//
//func newSnippetRenderer(c interface{}, before ...func()) chartrender.Renderer {
//	var baseTpl = `
//<div class="container">
//    <div class="item" id="{{ .ChartID }}" style="width:{{ .Initialization.Width }};height:{{ .Initialization.Height }};"></div>
//</div>
//{{- range .JSAssets.Values }}
//   <script src="{{ . }}"></script>
//{{- end }}
//<script type="text/javascript">
//    "use strict";
//    let goecharts_{{ .ChartID | safeJS }} = echarts.init(document.getElementById('{{ .ChartID | safeJS }}'), "{{ .Theme }}");
//    let option_{{ .ChartID | safeJS }} = {{ .JSON }};
//    goecharts_{{ .ChartID | safeJS }}.setOption(option_{{ .ChartID | safeJS }});
//    {{- range .JSFunctions.Fns }}
//    {{ . | safeJS }}
//    {{- end }}
//</script>
//`
//	return &snippetRenderer{c: c, before: before, baseTpl: baseTpl}
//}
//
//func (r *snippetRenderer) Render(w io.Writer) error {
//	content := r.RenderContent()
//	_, err := w.Write(content)
//	return err
//}
//
//func (r *snippetRenderer) RenderSnippet() ChartSnippet {
//	var snippet ChartSnippet
//	return snippet
//}
//
//func (r *snippetRenderer) RenderContent() []byte {
//	const tplName = "chart"
//	for _, fn := range r.before {
//		fn()
//	}
//
//	tpl := template.
//		Must(template.New(tplName).
//			Funcs(template.FuncMap{
//				"safeJS": func(s interface{}) template.JS {
//					return template.JS(fmt.Sprint(s))
//				},
//			}).
//			Parse(r.baseTpl),
//		)
//	var buf bytes.Buffer
//
//	err := tpl.ExecuteTemplate(&buf, tplName, r.c)
//
//	return buf.Bytes()
//}

// generate random data for bar chart
func generateBarItems() []opts.BarData {
	items := make([]opts.BarData, 0)
	for i := 0; i < 7; i++ {
		items = append(items, opts.BarData{Value: rand.Intn(300)})
	}
	return items
}

func (c *apiConfig) handlerCharts(w http.ResponseWriter, r *http.Request) {

	bar := charts.NewBar()
	bar.SetGlobalOptions(charts.WithXAxisOpts(opts.XAxis{
		Type:  "value",
		Scale: opts.Bool(true)}),
		charts.WithYAxisOpts(opts.YAxis{
			Type:  "value",
			Scale: opts.Bool(true)}),
	)

	// Put data into instance
	bar.SetXAxis([]string{"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"}).
		AddSeries("Category A", generateBarItems()).
		AddSeries("Category B", generateBarItems())
	// Where the magic happens
	//f, _ := os.Create("bar.html")
	//	bar.Render(w)
	//	bar.Renderer = newSnippetRenderer(bar, bar.Validate)
	//
	//	var buf bytes.Buffer
	//	err := bar.Render(&buf)
	//	if err != nil {
	//		log.Println(err)
	//		return
	//	}
	//

	res := bar.RenderSnippet()
	//	htmlchart := template.HTML(buf.String())
	fmt.Println("ELEMENT:", res.Element)
	fmt.Println("OPTION:", res.Option)
	fmt.Println("SCRIPT:", res.Script)

	resb := bar.RenderContent()
	fmt.Println(string(resb))
	tophtml := fmt.Sprintf(`<html>
		<head>
		<script src="https://go-echarts.github.io/go-echarts-assets/assets/echarts.min.js"></script>
		</head>
		<body>
		<center>`)

	w.Write([]byte(tophtml))

	w.Write([]byte("test"))
	w.Write([]byte(res.Element))
	w.Write([]byte(res.Script))

	bothtml := fmt.Sprintf(`
		</center>
		</body>
		</html>`)

	w.Write([]byte(bothtml))
}
