function buildTree(data) {
    margin = ({top: 20, right: 10, bottom: 10, left: 50})
    width = 200;
    height = 200;
    dx = 40;
    dy = 40;
    tree = d3.tree().nodeSize([dx, dy])
    diagonal = d3.linkVertical().x(d => d.x).y(d => d.y - 2)
    const root = d3.hierarchy(data);

    function initNode(d) {
        d.id = d.data.id;
        d._children = d.children;
        if (d.depth && (d.data.type == "real" || d.data.type == "model_init")) {
            d.children = null;
        }
    };

    root.x0 = width / 2;
    root.y0 = 0;
    root.descendants().forEach(initNode);

    const svg = d3.select(".canvas").append("svg")
        .attr("viewBox", [-margin.left, -margin.top, width, height])
        .style("font", "3px sans-serif")
        .style("user-select", "none")
        .attr("width", "100%")
        .attr("height", "100%")

    svg.append("svg:defs").selectAll("marker")
        .data(["real", "model"])      // Different link/path types can be defined here
        .enter().append("svg:marker")    // This section adds in the arrows
        .attr("id", String)
        .attr("viewBox", "0 -5 10 10")
        .attr("refX", d => d == "real" ? 15 : 10)
        .attr("refY", 0)
        .attr("markerWidth", 3)
        .attr("markerHeight", 3)
        .attr("orient", "auto")
        .attr("fill", "#555")
        .attr("fill-opacity", 0.4)
        .append("svg:path")
        .attr("d", "M0,-5L10,0L0,5");

    const gLink = svg.append("g")
        .attr("fill", "none")
        .attr("stroke", "#555")
        .attr("stroke-opacity", 0.4)
        .attr("stroke-width", 1.5);

    const gRealLink = svg.append("g")
        .attr("fill", "none")
        .attr("stroke", "#555")
        .attr("stroke-opacity", 0.4)
        .attr("stroke-width", 1.5);

    const gNode = svg.append("g")
        .attr("cursor", "pointer")
        .attr("pointer-events", "all");

    const gActionLabel = svg.append("g");

    const gRewardLabel = svg.append("g");

    const duration = d3.event && d3.event.altKey ? 2500 : 250;
    // Is it needed?
    const transition = svg.transition()
        .duration(duration)
        .attr("viewBox", [-margin.left, -margin.top, width, height])
        .tween("resize", window.ResizeObserver ? null : () => () => svg.dispatch("toggle"));

    let graph = true;
    d3.select(".toggle-graph")
        .on('click' , () => {
            graph = !graph;
            d3.selectAll(".graph").style("display", graph ? "block" : "none");
            d3.selectAll(".text").style("display", !graph ? "block" : "none");
        });

    let collapsed = true;
    function toggleSidebar() {
        collapsed = !collapsed;
        d3.select('.canvas').style('width', collapsed ? '75%' : '50%');
        d3.select('.sidebar').style('width', collapsed ? '25%' : '50%');
    }

    function color(frac) {
        const minHue = 240, maxHue=-80;
        const hue = ((frac * (maxHue - minHue)) + minHue);
        const saturation = 90 - Math.cos(frac * 2 * Math.PI) * 10;
        const lightness = 50 - Math.sin(frac * 2 * Math.PI) * 20;
        return "hsl(" + hue + "," + saturation + "%," + lightness + "%)";
    }

    function showHistogram(name, data) {
        const keys = Object.keys(data).sort();
        const values = keys.map(key => data[key]);

        const margin = {top: 10, right: 10, bottom: 10, left: 40},
            width = 300 - margin.left - margin.right,
            height = 160 - margin.top - margin.bottom;

        const yMin = Math.min(d3.min(values), 0),
              yMax = Math.max(d3.max(values), 0);

        const y = d3.scaleLinear()
            .domain([yMin, yMax])
            .range([height, 0])
            .nice();

        const x = d3.scaleBand()
            .domain(d3.range(keys.length))
            .rangeRound([0, width], .2);

        const yAxis = d3.axisLeft(y);

        d3.select(".info.graph").append("div").html(name);

        const svg = d3.select(".info.graph").append("svg")
            .style("background-color", "#eee")
            .attr("width", width + margin.left + margin.right)
            .attr("height", height + margin.top + margin.bottom)
          .append("g")
            .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

        svg.selectAll(".bar")
            .data(keys)
          .enter().append("rect")
            .style("fill", (k, i) => color(i / (keys.length - 1)))
            .attr("y", k => y(Math.max(0, data[k])))
            .attr("x", (k, i) => x(i))
            .attr("height", k => Math.abs(y(data[k]) - y(0)))
            .attr("width", x.bandwidth() - 2);

        svg.append("g")
            .attr("class", "x axis")
            .call(yAxis);

        svg.append("g")
            .attr("class", "y axis")
          .append("line")
            .attr("y1", y(0))
            .attr("y2", y(0))
            .attr("x1", 0)
            .attr("x2", width);
    }

    function showLegend(keys) {
        const legend = d3.selectAll(".legend").html("");
        keys.forEach((k, i) => {
            legend.append("div")
                .text(k + " ")
                .style("color", color(i / (keys.length - 1)));
        });
    }

    function update(source) {
      // Compute the new tree layout.
      tree(root);

      const nodes = root.descendants().reverse();
      const links = root.links();

      // Update the nodes…
      const node = gNode.selectAll("g")
        .data(nodes, d => d.id);

      // Enter any new nodes at the parent's previous position.
      const nodeEnter = node
          .enter()
          .append("g")
          .attr("transform", d =>
              d.data.type == "real" ? `translate(${d.x},${d.y})` :
              `translate(${source.x},${source.y})`
          )
          .attr("fill-opacity", d => d.data.type == "real" ? 1 : 0)
          .attr("stroke-opacity", d => d.data.type == "real" ? 1 : 0)
          .style("visibility", d => d.data.type == "root" ? "hidden" : "visible")
          .on("click", async d => {
            if(d.data.stub) {
                // Load data from the backend.
                const data = await d3.json("/entity/" + d.data.id);
                const node = d3.hierarchy(data);
                node.descendants().forEach(dd => {
                    initNode(dd);
                    dd.depth += d.depth;
                });
                d._children = node._children;
                d._children.forEach(dd => {
                    dd.parent = d;
                });
                d.data.stub = false;
            }

            d.children = d.children ? null : d._children;
            update(d);
          })
          .on("mouseover", showNodeInfo);

      nodeEnter.append("circle")
          .attr("r", 2.5)
          .attr("fill", d => d.data.children !== undefined ? "#555" : "#999")
          .attr("stroke-width", 10)
          .filter(d => d.data.terminal)
          .clone(true).lower()
          .attr("fill", "#fff")
          .attr("r", 3)
          .clone(true).lower()
          .attr("fill", d => "#999")
          .attr("r", 3.5)

      // Transition nodes to their new position.
      const nodeUpdate = node.merge(nodeEnter).transition(transition)
          .attr("transform", d => `translate(${d.x},${d.y})`)
          .attr("fill-opacity", 1)
          .attr("stroke-opacity", 1);

      // Transition exiting nodes to the parent's new position.
      const nodeExit = node.exit().transition(transition).remove()
          .attr("transform", d => `translate(${source.x},${source.y})`)
          .attr("fill-opacity", 0)
          .attr("stroke-opacity", 0);

      // Update the links…
      const link = gLink.selectAll("path")
        .data(links, d => 'l' + d.target.id);

      // Enter any new links at the parent's previous position.
      const linkEnter = link.enter()
          .filter(d => d.source.data.type != "root")
          .append("path")
          .attr("d", d => {
              const o = {x: source.x0, y: source.y0};
              return diagonal({source: o, target: o});
          })
          .attr("marker-end", "url(#model)")
          .on("mouseover", showLinkInfo);

      // Transition links to their new position.
      link.merge(linkEnter).transition(transition)
          .attr("d", diagonal);

      // Transition exiting nodes to the parent's new position.
      link.exit().transition(transition).remove()
          .attr("d", d => {
            const o = {x: source.x, y: source.y};
            return diagonal({source: o, target: o});
          });

      function addLabels(label, text, fill, opacity, dx, dy) {
          const labelEnter = label.enter()
               .append("text")
               .text(text)
               .attr("x", d => d.source.x)
               .attr("y", d => d.source.y);
          label.merge(labelEnter)
               .transition(transition)
               .attr("x", d => (d.source.x + d.target.x)/2 + dx)
               .attr("y", d => (d.source.y + d.target.y)/2 + dy)
               .attr("fill", fill)
               .attr("fill-opacity", opacity);
          label.exit()
               .transition(transition).remove()
               .attr("x", d => d.source.x)
               .attr("y", d => d.source.y);
      }

      function showInfo(stateId, info) {
          d3.select(".info.graph").html(
              "<img src=\"/render_state/" + stateId + "\">"
          );
          d3.select(".info.graph img").on('click' , toggleSidebar);
          d3.select(".info.text").html(
              "<pre>" + JSON.stringify(info, null, 2) + "</pre>"
          );
      }

      function showNodeInfo(d) {
          const info = {
              state_info: d.data.state_info,
              terminal: d.data.terminal,
          };
          showInfo(d.data.id, info);
      }

      function showLinkInfo(d) {
          const info = {
              agent_info: d.target.data.agent_info,
              action: d.target.data.action,
              reward: d.target.data.reward,
              from_state_info: d.source.data.state_info,
          };
          showInfo(d.source.data.id, info);
          const data = d.target.data.agent_info;
          let keys;
          Object.keys(data).sort().forEach(name => {
              if (data[name] instanceof Object) {
                  showHistogram(name, data[name]);
                  keys = Object.keys(data[name]).sort();
              }
          });
          if (keys) {
              showLegend(keys);
          }
      }

      const realTransitions = d3.zip(root.children.slice(0, -1), root.children.slice(1))
        .map(pair => ({source: pair[0], target: pair[1]}));

      const visibleLinks = d3.merge([
          links.filter(d => d.source.data.type != "root"),
          realTransitions
      ]);

      addLabels(
          gActionLabel.selectAll("text").data(visibleLinks, d => 'a' + d.target.id),
          d => d.target.data.action,
          d => "#fa0",
          d => 1,
          -10, -2
      );

      function renderReward(reward) {
          const prefix = reward > 0 ? '+' : '-';
          return prefix + Math.round(reward * 100) / 100;
      }

      addLabels(
          gRewardLabel.selectAll("text").data(visibleLinks, d => 'r' + d.target.id),
          d => renderReward(d.target.data.reward),
          d => d.target.data.reward > 0 ? "#0f0" : "#f00",
          d => d.target.data.reward ? 1 : 0,
          -10, 4
      );

      const realLink = gRealLink.selectAll("path")
        .data(realTransitions, d => 'l' + d.target.id);

      function straight(pair) {
          return "M" + pair.source.x + "," + pair.source.y + " " +
              pair.target.x + "," + pair.target.y;
      }

      const realLinkEnter = realLink
          .enter()
          .append("path")
          .attr("d", straight)
          .attr("marker-end", "url(#real)")
          .on("mouseover", showLinkInfo);

      realLink.merge(realLinkEnter)
          .transition(transition)
          .attr("d", straight);

      // Stash the old positions for transition.
      root.eachBefore(d => {
          d.x0 = d.x;
          d.y0 = d.y;
      });

      svg.call(d3.zoom()
          .extent([[0, 0], [width, height]])
          .scaleExtent([0.1, 10])
          .on("zoom", zoomed));

      function zoomed() {
          gNode.attr("transform", d3.event.transform);
          gLink.attr("transform", d3.event.transform);
          gRealLink.attr("transform", d3.event.transform);
          gActionLabel.attr("transform", d3.event.transform);
          gRewardLabel.attr("transform", d3.event.transform);
      }

      svg.call(d3.drag()
         .on("start", dragstarted)
         .on("drag", dragged)
         .on("end", dragended));

      function dragstarted() {
          d3.select(this).raise();
          g.attr("cursor", "grabbing");
      }

      function dragged(d) {
          d3.select(this).attr("cx", d.x = d3.event.x).attr("cy", d.y = d3.event.y);
      }

      function dragended() {
          g.attr("cursor", "grab");
      }
    }

    update(root);
}
