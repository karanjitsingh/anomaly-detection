Plotly.relayout(document.querySelectorAll("div")[1],   {margin: {l: 0,r: 0,b: 0,t: 0, pad: 0
  }}).then(() => Plotly.toImage(document.querySelectorAll("div")[1],{height:300,width:1000}).then(result => {
  var link = document.createElement('a');
  link.setAttribute('download', 'MintyPaper.png');
  link.setAttribute('href', result.replace("image/png", "image/octet-stream"));
  link.click();
}))