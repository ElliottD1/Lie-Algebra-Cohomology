
(function () {

  function getElByAny(...ids) {
    for (const id of ids) {
      const el = document.getElementById(id);
      if (el) return el;
    }
    return null;
  }


  const computeBtn = getElByAny("compute", "computeHasse", "computeHasseBtn");
  const cohoBtn = getElByAny("computeCohomology", "computeCohomologyBtn", "computeCoho");
  const groupEl = getElByAny("group");
  const nEl = getElByAny("n");
  const lEl = getElByAny("l");
  const selectedEl = getElByAny("selected");
  const adjointCheckbox = getElByAny("useAdjoint");
  const positiveCheckbox = getElByAny("positiveOnly");
  const resultsPre = getElByAny("results", "cohomology-output", "cohomologyResults", "cohomologyResultsBox");


  let results = resultsPre;
  if (!results) {
    results = document.createElement("pre");
    results.id = "results";
    document.body.appendChild(results);
  }


  function show(msg) {
    results.textContent = msg;
    console.log(msg);
  }


  function parseSelected(s) {
    if (!s) return [];
    if (Array.isArray(s)) return s.map(x => Number(x)).filter(x => !isNaN(x));
    return String(s).split(",").map(x => parseInt(x.trim())).filter(x => !isNaN(x));
  }

  async function postJSON(url, payload) {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const text = await res.text();
    let parsed;
    try { parsed = JSON.parse(text); } catch (e) { parsed = text; }
    if (!res.ok) {
      const msg = (parsed && parsed.detail) ? parsed.detail : (parsed && parsed.error) ? parsed.error : text;
      throw new Error(`${res.status} ${res.statusText} — ${msg}`);
    }
    return parsed;
  }


  function drawTree(tree) {
    try {
      const svg = d3.select("#diagram");
      if (svg.empty()) {
        d3.select("body").append("svg").attr("id", "diagram").attr("width", 900).attr("height", 600);
      }
      const s = d3.select("#diagram");
      s.selectAll("*").remove();

      const nodes = (tree && tree.nodes) ? tree.nodes : [];
      const edges = (tree && tree.edges) ? tree.edges : [];

      if (nodes.length === 0) {
        s.append("text").attr("x", 20).attr("y", 20).text("No nodes to display");
        return;
      }

      const depthMap = {};
      nodes.forEach(n => {
        if (!depthMap[n.depth]) depthMap[n.depth] = [];
        depthMap[n.depth].push(n);
      });

      const depths = Object.keys(depthMap).map(d => parseInt(d)).sort((a, b) => a - b);
      const width = parseInt(s.attr("width")) || 900;
      const height = parseInt(s.attr("height")) || 600;
      const levelHeight = height / (depths.length + 1);
      const positions = {};

      depths.forEach((depth, i) => {
        const ns = depthMap[depth];
        const y = (i + 1) * levelHeight;
        const spacing = width / (ns.length + 1);
        ns.forEach((node, j) => {
          const x = (j + 1) * spacing;
          positions[node.id] = { x, y };
          

          const vectorText = Array.isArray(node.vector) ? node.vector.join(",") : JSON.stringify(node.vector);
          s.append("text")
            .attr("x", x)
            .attr("y", y)
            .attr("text-anchor", "middle")
            .attr("class", "node-text")
            .attr("font-size", "12px")
            .attr("fill", "#000")
            .text(`[${vectorText}]`);
        });
      });

      edges.forEach(e => {
        const a = positions[e.source];
        const b = positions[e.target];
        if (!a || !b) return;
        s.append("line")
          .attr("x1", a.x).attr("y1", a.y + 10)
          .attr("x2", b.x).attr("y2", b.y - 10)
          .attr("stroke", "#333")
          .attr("stroke-width", 1);
        s.append("text")
          .attr("x", (a.x + b.x) / 2)
          .attr("y", (a.y + b.y) / 2)
          .attr("text-anchor", "middle")
          .attr("font-size", 10)
          .attr("fill", "#666")
          .text(e.move || "");
      });
    } catch (err) {
      console.error("drawTree error:", err);
      show("drawTree error: " + err.message);
    }
  }


  async function handleCompute() {
    try {
      const group = (groupEl && groupEl.value) ? groupEl.value : "A";
      const n = nEl ? parseInt(nEl.value) : 4;
      const l = lEl ? parseInt(lEl.value) : 2;
      const selected = selectedEl ? parseSelected(selectedEl.value) : [2];
      const payload = { n: Number(n), l: Number(l), selected };
      show("Computing Hasse tree...");

      const data = await postJSON(`/compute/${group}`, payload);
      if (!data || !data.tree) throw new Error("No tree returned");
      drawTree(data.tree);
      show(`Hasse tree computed — nodes: ${data.tree.nodes.length}`);
    } catch (err) {
      console.error(err);
      show("Compute error: " + (err.message || err));
    }
  }

  async function handleCohomology() {
    try {
      const group = (groupEl && groupEl.value) ? groupEl.value : "A";
      const n = nEl ? parseInt(nEl.value) : 4;
      const l = lEl ? parseInt(lEl.value) : 2;
      const selected = selectedEl ? parseSelected(selectedEl.value) : [2];
      const useAdjoint = adjointCheckbox ? adjointCheckbox.checked : false;
      const payload = { n: Number(n), l: Number(l), selected, use_adjoint: useAdjoint };
      console.log("Sending payload to backend:", JSON.stringify(payload));
      const data = await postJSON(`/cohomology/${group}`, payload);
      
      if (!data) throw new Error("No response data");
      const tree = data.tree || {};
      const results = data.cohomology_results || [];
      const gradations = data.gradations || [];
      const weightUsed = data.weight_used || [];

      drawTree(tree);
      const out = [];
      out.push(`=== COHOMOLOGY COMPUTATION ===`);
      out.push(`Weight used: [${weightUsed.join(", ")}]`);
      out.push(`Use adjoint: ${data.use_adjoint ? "YES" : "NO"}`);
      out.push(`Hasse nodes: ${tree.nodes ? tree.nodes.length : 0}`);
      out.push("");
      out.push(`=== COHOMOLOGY RESULTS ===`);
      
      if (results.length === 0) {
        out.push("(no results)");
      } else {
        results.forEach(r => {
          out.push(`Depth ${r.depth}, Path [${(r.path || []).join(",")}]`);
          out.push(`  Vector: [${(r.vector_at_node || []).join(",")}]`);
          const aff = r.affine_result || {};
          if (aff.error) {
            out.push(`  Error: ${aff.error}`);
          } else {
            out.push(`  Output: [${(aff.output || []).join(",")}]`);
          }
          out.push("");
        });
      }
      

      if (data.use_adjoint) {
        out.push(`=== GRADATIONS ===`);
        
        if (gradations.length === 0) {
          out.push("(no gradations computed)");
        } else {
          const positiveOnly = positiveCheckbox ? positiveCheckbox.checked : false;
          let filteredGradations = gradations;
          
          if (positiveOnly) {
            filteredGradations = gradations.filter(g => !g.error && g.gradation > 0);
          }
          
          if (filteredGradations.length === 0) {
            out.push("(no gradations to display)");
          } else {
            filteredGradations.forEach(g => {
              out.push(`Depth ${g.depth}, Path [${(g.path || []).join(",")}]`);
              out.push(`  Vector: [${(g.vector || []).join(",")}]`);
              out.push(`  Output weight: [${(g.output_weight || []).join(",")}]`);
              

              const formattedRootCoords = (g.root_coordinates || []).map(x => {
                const formatted = x.toFixed(4);
                return formatted === "-0.0000" ? "0.0000" : formatted;
              }).join(", ");
              out.push(`  Root coordinates: [${formattedRootCoords}]`);
              out.push(`  Active roots: [${(g.active_roots || []).join(", ")}]`);

              let gradStr = "N/A";
              if (g.gradation !== undefined && g.gradation !== null) {
                const val = g.gradation;
                const formatted = val.toFixed(4);
                gradStr = formatted === "-0.0000" ? "0.0000" : formatted;
              }
              out.push(`  Gradation: ${gradStr}`);
              out.push("");
            });
          }
        }
      }
      
      show(out.join("\n"));
    } catch (err) {
      console.error(err);
      show("Cohomology error: " + (err.message || err));
    }
  }


  if (computeBtn) computeBtn.addEventListener("click", handleCompute);
  if (cohoBtn) cohoBtn.addEventListener("click", handleCohomology);


  if (typeof d3 === "undefined") console.warn("D3 is not loaded. Hasse diagram drawing will fail.");

})();