from agent_collaboration import run_agents

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("   🛒  RetailInvenAI  —  Multi-Agent Intelligence System")
    print("=" * 60)
    print("\nChoose Mode:")
    print("  1. CSV Mode      (batch-process all products from CSVs)")
    print("  2. Manual Mode   (single-product interactive input)")

    choice = input("\nEnter choice (1 or 2): ").strip()

    if choice == "2":
        # ── MANUAL INPUT MODE ─────────────────────────────────────────────
        price      = float(input("Enter Price: "))
        promo      = input("Promotion (Yes/No): ")
        trend      = input("Trend (Increasing/Stable/Decreasing): ")

        stock    = int(input("Stock Level: "))
        lead     = int(input("Lead Time (days): "))
        reorder  = int(input("Reorder Point: "))

        comp_price = float(input("Competitor Price: "))
        discount   = float(input("Discount: "))
        elasticity = float(input("Elasticity Index: "))

        result = run_agents(
            price, promo, trend,
            stock, lead, reorder,
            comp_price, discount, elasticity
        )

        print("\n" + "=" * 60)
        print("   AGENT RESULTS")
        print("=" * 60)
        print(f"  Demand Prediction  : {result['Demand']} units")
        print(f"  Stock Risk         : {result['Stock Risk']} "
              f"({result['Risk Probability']:.1f}%)")
        print(f"  Optimal Price      : ${result['Optimal Price']}")
        print(f"  Expected Revenue   : ${result['Max Revenue']}")
        print("\n" + "─" * 60)
        print("  AI DECISION")
        print("─" * 60)
        print(result["AI Decision"])

    else:
        # ── CSV BATCH MODE ────────────────────────────────────────────────
        num_str    = input("Enter number of products to process : ")
        num_records = int(num_str) if num_str.strip().isdigit() else 5

        # run_agents() will:
        #   1. Print each agent's results + charts
        #   2. Print the combined FUSION RESULT table
        #   3. Show AI strategies
        #   4. Show the fusion chart
        fusion_output = run_agents(num_records=num_records)

        print("\n" + "=" * 60)
        print("   ✅  All done!  Check the saved files:")
        print("      • agent_collaboration_output.csv")
        print("      • demand_agent_charts.png")
        print("      • inventory_agent_charts.png")
        print("      • pricing_agent_charts.png")
        print("      • fusion_result_summary.png")
        print("=" * 60)
