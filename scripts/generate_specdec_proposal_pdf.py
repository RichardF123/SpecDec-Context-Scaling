from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "pdf" / "specdec_frozen_submission_proposal_3members.pdf"


def make_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="TitleCentered",
            parent=styles["Title"],
            fontName="Times-Bold",
            fontSize=15,
            leading=18,
            alignment=TA_CENTER,
            spaceAfter=6,
            textColor=colors.HexColor("#111111"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="TeamLine",
            parent=styles["BodyText"],
            fontName="Times-Roman",
            fontSize=9.4,
            leading=11.5,
            alignment=TA_CENTER,
            spaceAfter=0,
            textColor=colors.HexColor("#111111"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionHead",
            parent=styles["Heading2"],
            fontName="Times-Bold",
            fontSize=10.7,
            leading=12.6,
            spaceBefore=7,
            spaceAfter=3,
            textColor=colors.HexColor("#111111"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyTextProposal",
            parent=styles["BodyText"],
            fontName="Times-Roman",
            fontSize=9.25,
            leading=11.15,
            spaceAfter=4.7,
            alignment=TA_LEFT,
        )
    )
    styles.add(
        ParagraphStyle(
            name="NameHead",
            parent=styles["BodyTextProposal"],
            fontName="Times-Bold",
            fontSize=9.25,
            leading=11.15,
            spaceBefore=2,
            spaceAfter=1.8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ContributionBody",
            parent=styles["BodyTextProposal"],
            fontSize=8.85,
            leading=10.35,
            spaceAfter=3.2,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ContributionName",
            parent=styles["NameHead"],
            fontSize=8.95,
            leading=10.35,
            spaceBefore=1.2,
            spaceAfter=1.0,
        )
    )
    return styles


def build_story(styles):
    story = []

    def p(text: str, style: str = "BodyTextProposal"):
        story.append(Paragraph(text, styles[style]))

    p(
        "When Does Speculative Decoding Stop Paying Off?<br/>"
        "Context Length and FP8-Compressed KV Cache on an 8GB GPU",
        "TitleCentered",
    )
    p(
        "Fang Chuming 23097386D&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
        "Deng Xiyue 23110509D&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
        "Cheng Shuyue 23097089D",
        "TeamLine",
    )
    story.append(Spacer(1, 7))

    p("1. Research Question & Motivation", "SectionHead")
    p(
        "This project studies the question: <b>How does context length affect "
        "speculative-decoding speedup on an 8GB consumer GPU, and does "
        "FP8-compressed KV cache change the context range in which speculative "
        "decoding remains beneficial?</b> It also examines whether any observed "
        "degradation is mainly associated with reduced accepted-token benefit or "
        "increased draft/verification overhead."
    )
    p(
        "Speculative decoding can reduce generation latency by using a smaller "
        "draft model to propose several tokens and a larger target model to verify "
        "them. Its practical benefit may depend strongly on context length. As "
        "context grows, KV-cache memory usage and verification cost increase, which "
        "may reduce the advantage of accepting multiple draft tokens per "
        "target-model step. This issue is especially relevant for small open-weight "
        "models running on consumer GPUs, where memory pressure can determine "
        "whether long-context inference is feasible. The project is an Efficient "
        "LLM Inference empirical systems evaluation focused on controlled "
        "measurement of context length, FP8-compressed KV cache, and speculative "
        "decoding under an 8GB GPU constraint."
    )

    p("2. Method / System", "SectionHead")
    p(
        "The system uses a custom PyTorch eager inference engine for Qwen2.5 "
        "models. The target model is <b>Qwen2.5-1.5B-Instruct</b>, and the "
        "draft/development model is <b>Qwen2.5-0.5B-Instruct</b>. Existing "
        "development-stage validation on the 0.5B model includes full-forward "
        "correctness validation, KV-cache correctness validation, greedy generation "
        "correctness, chunked prefill implementation, 4K chunked-prefill "
        "equivalence, identification of the memory cause of raw 8K eager-prefill "
        "out-of-memory behavior, and a baseline sanity comparison against "
        "transformers eager. A Hugging Face Transformers eager implementation will "
        "also be used as an external sanity reference to check that the custom "
        "engine operates in a reasonable performance range."
    )
    p(
        "Formal benchmarking will start after the corresponding correctness checks "
        "are completed for the 1.5B target model. The tested system includes KV "
        "caching, chunked prefill for long contexts, unquantized BF16 KV cache, "
        "FP8-compressed KV cache, standard autoregressive decoding, and fixed-chain speculative "
        "decoding with proposal length <b>k = 4</b> under greedy generation. "
        "FP8-compressed KV refers to storing KV cache tensors in compressed FP8 "
        "form and dequantizing them for eager attention, rather than implementing "
        "native FP8 attention."
    )

    p("3. Experimental Design & Evaluation", "SectionHead")
    p(
        "The central experiment is a <b>3 x 2 x 2 controlled factorial design</b> "
        "with three factors: context length, KV representation, and decoding "
        "method. Context length will be approximately 2K, 4K, and 8K tokens. KV "
        "representation will be FP16 or FP8-compressed. Decoding will be either "
        "standard autoregressive decoding or speculative decoding. This produces "
        "12 experimental conditions."
    )
    p(
        "Prompts will be constructed as paired prompt families. For each family, "
        "the 2K, 4K, and 8K contexts will be nested prefixes of the same base text, "
        "and the continuation task will remain the same across lengths. Contexts "
        "will target approximately 2K, 4K, and 8K tokens with a length tolerance of "
        "at most 32 tokens; actual token counts will be recorded and no synthetic "
        "filler will be inserted. A pilot run will use 4 prompt families to check "
        "feasibility, instrumentation, timing stability, and correctness. The "
        "formal experiment will use at least 16 prompt families."
    )
    p(
        "Performance evaluation will report decode latency, end-to-end latency, "
        "tokens per second, speculative speedup, and peak VRAM. Decode-only "
        "speculative speedup is the primary performance metric. End-to-end speedup "
        "is secondary and includes target and draft prefill costs. Speedup "
        "comparisons will stay within the same KV representation: FP16 speculative "
        "decoding will be compared with FP16 autoregressive decoding, and FP8 "
        "speculative decoding will be compared with FP8 autoregressive decoding. "
        "Each condition will be measured over repeated runs, with performance "
        "summarized across prompt families rather than using best-case timing."
    )
    p(
        "Speculative-decoding behavior will be measured using acceptance rate, "
        "accepted tokens per verification, draft latency, and verification latency. "
        "These metrics support empirical attribution of speedup changes across "
        "context lengths. The analysis will examine whether long-context "
        "degradation is more closely associated with accepted-token benefit, draft "
        "overhead, or verification overhead. KV bandwidth may be used as supporting "
        "interpretation, but it will not be treated as an independently isolated "
        "measured component."
    )
    p(
        "FP8-compressed KV quality will be evaluated on the WikiText-2 test set "
        "using at least 32 fixed, non-overlapping windows per condition, with "
        "perplexity and Delta PPL reported relative to the unquantized BF16 KV "
        "baseline. Optional logit or top-1 sanity checks may also be used if useful. Perplexity is used to "
        "assess quality degradation from FP8 KV storage, not speculative-decoding "
        "performance. If speculative speedup crosses 1x between tested context "
        "lengths, one or two intermediate context lengths may be evaluated to "
        "estimate the break-even region."
    )

    p("4. Feasibility & Expected Outcomes", "SectionHead")
    p(
        "The project will run on an <b>NVIDIA RTX 4060 Laptop GPU with 8GB "
        "VRAM</b>. It uses small open-weight models and an already functioning "
        "custom inference engine, keeping the required compute resources within "
        "the scale of a course project. No model training or paid API usage is "
        "required; the main computational cost is local inference and benchmarking "
        "on a single RTX 4060 Laptop GPU. The remaining implementation, "
        "benchmarking, and analysis effort is estimated at approximately 140-175 "
        "person-hours across the three team members. The remaining core work "
        "includes 1.5B target correctness validation, formal 8K operational "
        "validation, FP8-compressed KV implementation, speculative-decoding "
        "implementation, instrumentation, pilot experiments, the final factorial "
        "experiment, and analysis."
    )
    p(
        "The scope is restricted to protect the quality of evaluation and "
        "interpretation. Optional CUDA-kernel work and weight quantization are not "
        "part of the committed core project. The experiments will examine whether "
        "the memory savings from FP8-compressed KV translate into practical latency "
        "improvements and whether they shift the context range in which speculative "
        "decoding remains beneficial. Negative or mixed results would still be "
        "informative by clarifying the conditions under which FP8-compressed KV or "
        "speculative decoding provides limited practical benefit."
    )

    p("5. Team Contributions", "SectionHead")
    p("Fang Chuming", "ContributionName")
    p(
        "Fang Chuming will take primary responsibility for custom inference-engine "
        "maintenance, Qwen2.5-1.5B target correctness and regression validation, "
        "KV-cache correctness, chunked-prefill integration, asymmetric V-only FP8 "
        "KV maintenance, FP8 scaling/dequantization and memory accounting, VRAM "
        "profiling, and KV-quality evaluation. Fang Chuming will also support "
        "system integration, reproducibility checks, failure diagnosis, and the "
        "final report.",
        "ContributionBody",
    )
    p("Deng Xiyue", "ContributionName")
    p(
        "Deng Xiyue will take primary responsibility for the fixed-chain "
        "speculative-decoding implementation, target/draft cache synchronization, "
        "cache trimming and rollback, block-verification alignment, "
        "speculative-decoding correctness validation including cache synchronization, "
        "verification alignment, and output equivalence, acceptance instrumentation, and decode and end-to-end "
        "performance measurement. Deng Xiyue will also support experiment "
        "automation, failure diagnosis, reproducibility checks, and quantitative "
        "analysis.",
        "ContributionBody",
    )
    p("Cheng Shuyue", "ContributionName")
    p(
        "Cheng Shuyue will take primary responsibility for paired prompt-family "
        "construction, pilot and final factorial experiment execution, experiment "
        "instrumentation and results-schema implementation, experiment configuration "
        "and results validation, repeated-run consistency checks, "
        "external-reference benchmarking, statistical aggregation, break-even "
        "analysis, figures and tables, and coordination of report and presentation "
        "materials. Cheng Shuyue will also independently review correctness "
        "evidence and contribute to reproducibility and failure-case analysis.",
        "ContributionBody",
    )
    p("Joint responsibilities", "ContributionName")
    p(
        "All three members will jointly finalize the experimental protocol, review "
        "code and formal results, interpret negative or mixed findings, prepare the "
        "final report, and prepare the presentation.",
        "ContributionBody",
    )

    return story


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Times-Roman", 8)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.drawRightString(letter[0] - 0.62 * inch, 0.42 * inch, f"Page {doc.page}")
    canvas.restoreState()


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    styles = make_styles()
    pdf = SimpleDocTemplate(
        str(OUT),
        pagesize=letter,
        rightMargin=0.62 * inch,
        leftMargin=0.62 * inch,
        topMargin=0.58 * inch,
        bottomMargin=0.58 * inch,
    )
    pdf.build(build_story(styles), onFirstPage=footer, onLaterPages=footer)
    print(OUT)


if __name__ == "__main__":
    main()
