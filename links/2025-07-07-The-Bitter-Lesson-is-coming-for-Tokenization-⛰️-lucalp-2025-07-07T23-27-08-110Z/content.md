  The Bitter Lesson is coming for Tokenization | ⛰️ lucalp                    

[

# ⛰️ lucalp

](/)

[Home](/) [Blog](/blog/) [Bio](/bio/) [Twitter](https://x.com/lucalp__)

# The Bitter Lesson is coming for Tokenization

*24 Jun, 2025*

*a world of LLMs without tokenization is desirable and increasingly possible*

Published on 24/06/2025 • ⏱️ 29 min read

* * *

> In this post, we highlight the desire to replace tokenization with a general method that better leverages compute and data. We'll see tokenization's role, its fragility and we'll build a case for removing it. After understanding the design space, we'll explore the potential impacts of a recent promising candidate (Byte Latent Transformer) and build strong intuitions around new core mechanics.

As it's been pointed out countless times - if the trend of ML research could be summarised, it'd be the adherence to [The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) - opt for general-purpose methods that leverage large amounts of compute and data over crafted methods by domain experts. More succinctly articulated by [Ilya Sutskever](https://x.com/ilyasut), "the models, they just want to learn". Model ability has continued to be blessed with the talent influx, hardware upgrades, model architectural advances and initial data ubiquity to enable this reality in recent years.

## the pervasive tokenization problem

However, one of the [documented bottlenecks](https://arxiv.org/abs/2310.08754) in the text transformer world that has received less optimisation effort is the very mechanism that shapes its world view - tokenization.

If you're not aware, one of the popular text tokenization methods for transformers, Byte-Pair Encoding (BPE), is [a learned procedure](https://huggingface.co/learn/llm-course/en/chapter6/5) that extracts an *effectively compressed* vocabulary (of desired size) from a dataset by iteratively merging the most frequent pairs of existing tokens.

[source](https://huggingface.co/learn/llm-course/en/chapter6/5)

It's worth remembering that this form of tokenization is *not* a strict requirement of the transformer. In practice, it means that we're able to represent more bytes given a fixed number of entries in the transformer's embedding table. From our earlier definition, *effective* is doing some heavy lifting. Ideally, the vocabulary of tokens is perfectly constructed for the task at hand such that it obtains the optimal trade off of byte compression to reduce the transformer's FLOPS while maintaining enough of a granular representation to achieve the lowest possible loss. Another ideal attribute is that tokens that do get merged, end up being well-modelled during training.

![](https://bear-images.sfo2.cdn.digitaloceanspaces.com/lucalp/tokenizer-map.webp)

*A crude map of sequence vs vocab size*

In the optimal tradeoff, the need for byte compression comes from attention's [computational complexity](https://proceedings.mlr.press/v201/duman-keles23a.html) and it's the core reason why transformers have to rely on some form of tokenization (often sub-word). Character-level RNNs used to be the norm ([Sutskever, 2011](https://icml.cc/2011/papers/524_icmlpaper.pdf), [Graves, 2013](https://arxiv.org/abs/1308.0850), [Karpathy's RNN post](https://karpathy.github.io/2015/05/21/rnn-effectiveness/#:~:text=use%20an%20LSTM.-,Character%2DLevel%20Language%20Models,-Okay%2C%20so%20we)) but they struggled to learn from characters and were superseded in favour of character-aware models that tokenize via [a CNN over characters](https://arxiv.org/abs/1508.06615) (which also spilled over to [transformer world](https://arxiv.org/abs/2010.10392)). In the case of attention however, tokenization was there [from the beginning](https://arxiv.org/html/1706.03762v7#:~:text=a%2032000%20word%2Dpiece%20vocabulary) since it is imperative to avoid clogging up the context to enable the transformer to attend to the full sequence and cash in on its [long-range dependencies abilities](https://arxiv.org/html/1706.03762v7#:~:text=The%20shorter%20these%20paths%20between%20any%20combination%20of%20positions%20in%20the%20input%20and%20output%20sequences%2C%20the%20easier%20it%20is%20to%20learn%20long%2Drange%20dependencies).

Revisiting the optimal tradeoff, tokenizers are often far from the ideal and the history of LLMs is plagued with downstream issues attributable to them. From these "earlier" days as modern LLMs started seeing more activity, [we saw things](https://www.youtube.com/watch?v=zduSFxRajkE) like:

*   a reddit user "SolidGoldMagikarp" getting [their own dedicated token](https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation) in OpenAI's tokenizer that was poorly modelled, eliciting the phenomena of ["glitch tokens"](https://arxiv.org/pdf/2402.14020)
*   GPT2's Python performance being worse than expected partially due to the way in spaces were tokenized ([paste in & see](https://tiktokenizer.vercel.app/?model=gpt2))
*   inability to detect the number of [r's in 🍓 meme](https://letmegooglethat.com/?q=how+many+r%27s+in+strawberry+meme)
*   numbers being [tokenized totally incoherently in GPT2](https://www.beren.io/2023-02-04-Integer-tokenization-is-insane/) which got rectified [a few different ways](https://www.beren.io/2024-05-11-Integer-tokenization-is-now-much-less-insane/) but the jury is [still out](https://arxiv.org/abs/2402.14903) on [what the consensus will be](https://huggingface.co/spaces/huggingface/number-tokenization-blog):

![](https://bear-images.sfo2.cdn.digitaloceanspaces.com/lucalp/gpt2_number_composition.webp)

*GPT2's number tokenization*

In most of these cases, the tokenizer & [its pipeline](https://huggingface.co/learn/llm-course/en/chapter6/4) gets [tweaked](https://arxiv.org/pdf/2402.01035) and the problems resolved. There've even been efforts to automatically detect [under-trained tokens](https://arxiv.org/abs/2405.05417) for glitch tokens. But examples like the 🍓 meme (and more later in the post) are more fundamental examples of how we're depriving models of information in the name of efficiency via simplistic levers.

For the purpose of this post, we'll limit our exploration to text tokenization but I would be remiss if I didn't mention that tokenization is a feature across all modalities with [modality-specific](https://arxiv.org/abs/2210.13438) [tokenizers](https://www.arxiv.org/abs/2412.13061) becoming the standard. This comes with its own host of challenges but continues to perpetuate the externalised, separately trained models that [have competing concerns](https://arxiv.org/html/2504.08736v1#:~:text=We%20identify%20the%20growing%20complexity%20of%20latent%20space%20as%20the%20key%20factor%20behind%20the%20reconstruction%20vs.%20generation%20dilemma.%20To%20mitigate%20this%2C%20we%20propose%20semantic%20regularization%2C%20which%20aligns%20tokenizer%20features%20with%20semantically%20consistent%20features%20from%20a%20pre%2Dtrained%20visual%20encoder.) and their own [training dynamics](https://arxiv.org/abs/2005.08520) which also end up having to be addressed [incrementally](https://arxiv.org/pdf/2005.00341), [via extension](https://arxiv.org/pdf/2306.06546) or via [improved approaches](https://arxiv.org/pdf/2309.15505). This is all to say, the problem is evidently non-trivial and has received significant research effort.

In the world of text tokenization, at least from an external point of view (though, not sure about [internally](https://x.com/Dorialexander/status/1913121809110626762)), things do seem to have stagnated. Even with this stability, the failure modes of tokenization continue to impede the models. A reasonable question to ask might be - "we have approaches that let us cope with these failure modes, do we really need to solve it?"

### can we just ignore it?

From earlier days, chain of thought, tool use and RAG all began addressing these issues and more recently, increasingly sophisticated [undisclosed mid/post-training recipes](https://arxiv.org/html/2412.15450v1#:~:text=with%20Phi%203.5%20incorporating%20multilingual%20data%20during%20mid%2Dtraining.%20However%2C%20the%20exact%20composition%20of%20languages%20in%20the%20training%20corpus%20is%20not%20disclosed%20%E2%80%93%20in%20fact%2C%20no%20languages%20are%20explicitly%20listed%2C%20resulting%20in%20a%20lack%20of%20transparency%20regarding%20the%20data%20sources%20and%20training%20procedures.) and the move to reasoning-based models continue in this direction. But it begs the question - how much model ability is being left on the table due to poor tokenization? In my view, this includes both [sub-optimal merges](https://arxiv.org/abs/2305.15425) for task diversity to misconfiguring the tokenizer [relative to model capacity](https://arxiv.org/html/2501.16975v2#:~:text=Using%20a%20large%20input%20vocabulary%2C%20we%20achieve%20performance%20comparable%20to%20double%2Dsized%20baselines%20with%20no%20additional%20cost). The honest answer here is that no one seems to have publicly investigated this thoroughly (from what I could find). However, the revealed preference of the big labs is in favour of subword-level tokenization and hasn't seen much movement. Given no direct research to consider, we'll use the latest in learned tokenization and byte-level end-to-end learned tokenization to be proxies for understanding what's being left on the table.

While researching for this post, I ended up reading a bit too much into [the text tokenization literature](https://arxiv.org/pdf/2112.10508) which probably warrants its own post. For the curious reader, [this provides a great overview and kick off point](https://arxiv.org/pdf/2112.10508) but in the interest of my own sanity and your time, just [trust me bro](https://bear-images.sfo2.cdn.digitaloceanspaces.com/lucalp/trust-me-bro.webp), there's quite a bit to it!

### can we just delete it?

Before attempting optimising, we should always ask the important question "[can we just delete it?](https://www.google.co.uk/books/edition/Elon_Musk/HjyvEAAAQBAJ?hl=en&gbpv=1&bsq=delete%20what%20you%20can&printsec=frontcover)". From a domain point of view, some [are skeptical](https://arxiv.org/pdf/2112.10508) that bytes are adequate for modelling natural language. However, if we only entertain the technical feasibility - what does deletion look like?

In [the GPT-2 paper](https://arxiv.org/pdf/1808.04444), the authors revisit the choice of input representation and, empirically, register a similar performance gap on WebText to Google's work in the [character-level LM with Deeper Self-Attention](https://arxiv.org/abs/1808.04444) paper. It kicked off the character-level revival by showing that, with the help of auxiliary losses, it outperformed its LSTM character-level counter parts but still registered a gap versus word-level models. The authors follow up with another paper that [bridges the gap](https://arxiv.org/pdf/1908.10322) but at the expense of much more compute and time to train.

This is all to say, we started at the character-level, authors tried going back to it but failed for other reasons that may encourage revision due to shifting underlying factors. If we look at BPE more closely, a commonly cited heuristic is that BPE tokens represent, on average, 4.4 bytes per token meaning that current BPE-based transformers' 32K token context windows are able to attend over ~140K bytes with a vocab size of 256K. If we were to use pure bytes and a vanilla transformer modelling UTF-8 bytes, we'd have a vocab size of 256 and be limited to attending to only 32K bytes!

So Google's [ByT5](https://arxiv.org/abs/2105.13626) set out to answer the "can we delete?" question in its purest form:

> Our goal in designing ByT5 is to take an existing token-based model and perform the minimal set of modifications to make it token-free, thereby limiting experimental confounds.

They showed that pure byte modelling, even when trained on 4x less data, had comparable or better performance to its SentencePiece counter part on a subset of benchmarks under 1B parameters (namely robustness to noise, word-level tasks like transliteration, morphological inflection, graphene-to-phoneme[1](#fn-1)). Given the intentionally naive modification, it increased pre-training time by 33% (wall-clock time) and in the worst case[2](#fn-2), inference by up to a factor of 10x[3](#fn-3).

However if one were to, hypothetically, entertain the heretical thoughts of straying from The One True Architecture then one could be free of attention's quadratic complexity and worry less about clogging up the context.

![alternative architecture slander as a meme](https://bear-images.sfo2.cdn.digitaloceanspaces.com/lucalp/heretic.webp)

*low effort alternative architecture slander*

In that case, ByT5's kindred soul (w.r.t simplicity) is [MambaByte](https://arxiv.org/abs/2401.13660) that capitalises on [State Space Model's](https://arxiv.org/abs/2111.00396) (SSM) fixed size memory state that doesn't scale with input context size which, when not dealing with compressed byte representations via subword-level tokenization, becomes a great fit for the problem. However even without the clogged context problem, the sequence length still remains and so do the increased inference steps so they employ the model in a speculative decoding setup to alleviate the burden. SSMs are [a tool in the kit](https://www.arxiv.org/pdf/2412.11084) that [have found strong utility](https://arxiv.org/abs/2306.15794) where MambaByte would be useful but [they come](https://openreview.net/pdf?id=pymXpl4qvi) with their [own host of challenges](https://arxiv.org/pdf/2505.15105) that [we inherit](http://kempnerinstitute.harvard.edu/research/deeper-learning/repeat-after-me-transformers-are-better-than-state-space-models-at-copying/) when relying on them as the core method to remove the tokenizer.

Alas, given we are True Believers we would never have such thoughts. We hold steady faith in the Values of The Transformer and thus [heretics we are not](https://bear-images.sfo2.cdn.digitaloceanspaces.com/lucalp/god.webp).

## so... can we *learn* it?

In comparison to BPE's learning, there's a series of architecture changes we can make to a transformer to remove the requirement of optimised sub-word tokenization. With the bitter lesson in mind, if we're able to learn tokenization more generally, we would expect to see a model:

1.  be competitive or improve loss scores
2.  improve on downstream tasks across the board
3.  demonstrate better scaling curves when thrown more compute and data

Before jumping into transformer modifications - are there any directionally relevant changes we can make to vanilla BPE? Mostly, they're incremental changes to compensate for its limitations but aren't aligned with our previously stated goal. It includes things like [probabilistically skipping merge operations](https://arxiv.org/pdf/1910.13267) (common in [a variety](https://proceedings.mlr.press/v139/ramesh21a.html?ref=journey) of [tasks](https://proceedings.mlr.press/v202/radford23a/radford23a.pdf)), a pretokenization curriculum to first learn subwords then [super words that bridge whitespace](https://arxiv.org/abs/2503.13423), falling back to bytes instead of lumping everything into the `<unk>` token and enforcing [consistency of predictions](https://arxiv.org/pdf/2103.08490) over different segmentations. However methods like updating the tokenizer based on [downstream loss under different segmentations](https://aclanthology.org/2020.findings-emnlp.120/) and jointly optimizing the [tokenizer with the model](https://aclanthology.org/2021.findings-acl.21/) are more aligned with our goal but are trickier to apply in practice.

Given we're seeking generality that demonstrates better scaling curves, this isn't going to cut it.

### design space so far

Rather than going back [three decades](https://proceedings.neurips.cc/paper/1995/file/c667d53acd899a97a85de0c201ba99be-Paper.pdf) to paint a deep picture of the space, we'll focus in on the recent progress in the transformer-centric literature where there's been a few different stabs at addressing the efficiency challenges of pure byte modelling for the transformer case.

Each architecture is some variation on the theme of creating a compressed representation which usually materialise in a few choices:

1.  down/upsampling to/from that compressed representation
2.  how FLOPS are distributed across levels of representation
3.  decoding strategy [4](#fn-4)
4.  fixed or dynamic width of bytes

In language modelling, it's commonplace to compare perplexity but in papers like these where we're not evaluating with a fixed tokenizer, some variation of bits-per-byte will be used as a tokenizer independent version of perplexity:

BPB(x)\=ℒCE(x)ln(2)·nbytes 

Without getting bogged down in excessive detail, let's consider some directionally-aligned landmark papers in recent memory.

[CANINE](https://arxiv.org/abs/2103.06874)'s encoder (targeting non-generative tasks) used a combination of n-gram hash embeddings, local attention and strided convolutions to downsample from character-level to a compressed representation processable by a larger transformer [5](#fn-5).

![](https://bear-images.sfo2.cdn.digitaloceanspaces.com/lucalp/canine.webp)

[Charformer](https://arxiv.org/abs/2106.12672) is an encoder-decoder model that also learns to downsample end-to-end via a gradient-based block scoring function up to some fixed block size[6](#fn-6). It isn't designed to be autoregressive either.

Concretely, from the character sequence it builds byte embeddings from which it constructs a series of *candidate* latent subword blocks up to a max block size at some stride. Stride size is set to the size of block for that block size.

![](https://bear-images.sfo2.cdn.digitaloceanspaces.com/lucalp/charformer1.webp)

At each position, which latent subword block should we use? This is enabled by a block scoring network to select the right block which gives us a score per block for each position i. Scores are then softmax'd to get a probability distribution Pi over blocks for position i. These subword block representations are summed and weighted by their Pb,i to form the final latent subword representation for position i:

X^i\=∑bMPb,iXb,i

And visually[7](#fn-7):

![](https://bear-images.sfo2.cdn.digitaloceanspaces.com/lucalp/charformer2.webp)

In its original form (like CANINE), it can't be used in an autoregressive setting due to the downsampling for block scoring since no mask can be applied to ensure no subword is formed with future bytes at each position.

Building off of this, the [Hourglass Transformers](https://arxiv.org/abs/2110.13711) paper is a [U-Net-like](https://arxiv.org/abs/1505.04597) architecture that shows the success of adapting an autoregressive transformer with downsampling by some static factor at different stages (hence hierarchy in paper title) followed by upsampling with residual connections from the pre-pooled representation[8](#fn-8). The down/upsampling are attention-based where they down/upsample the attention's queries via some arbitrary function (respectively average pool, linear upsampling).

![](https://bear-images.sfo2.cdn.digitaloceanspaces.com/lucalp/hourglass1.webp)

Given they're partially targeting the task of language modelling, they resolve the information leak problem by doing an additional patch-aware shifting of labels to preserve the autoregressive property of the model. They also conduct interesting ablations such as scaling the intermediate layers acting on the downsampled sequence representation (thematically relevant). Crucially though, each time a token is decoded, the entire new sequence has to be passed through the entire network.

They show that they're able to improve upon a baseline byte-level model while still reducing the total number of parameters (at the 150M scale):

![](https://bear-images.sfo2.cdn.digitaloceanspaces.com/lucalp/hoursglass2.webp)

One of the primary authors extends this architecture and goes on to experiment in [Efficient Transformers with Dynamic Token Pooling](https://arxiv.org/abs/2211.09761) to replace the static patching. Given the patch boundary is to be dynamic, they experiment with learning a boundary predictor during training via:

1.  supervision via tokenizer (ala CANINE-S[5](#fn-5))
2.  supervision via spikes in the conditional entropy of the predictive distribution
3.  end-to-end via stochastic re-parametrisation

They also experiment with not learning the boundary predictor and just relying on a modality-specific boundary via the whitespace character.

Just as the previous architecture, it still requires the full model (i.e boundary predictor network, token model and the decoder) to be invoked after each character is decoded[9](#fn-9).

[MEGABYTE](https://arxiv.org/abs/2305.07185)'s multiscale transformers is the next autoregressive approach to look at:

![](https://bear-images.sfo2.cdn.digitaloceanspaces.com/lucalp/megabyte.webp)

It downsamples from the byte-level by embedding each position in the byte-sequence and chunking it into static patches of length P and then employing multiscale transformers to model the patch-level and byte-level sequences. If you haven't come across the term "multiscale", much like "hierarchical" in the previous paper, it isn't [a new term](https://proceedings.neurips.cc/paper/1995/file/c667d53acd899a97a85de0c201ba99be-Paper.pdf) and it's used here to refer to a large global model that functions on a compressed sequence (i.e patch-level) and a small local model that operates on the full sequence (i.e byte-level). After reviewing the past few papers, hopefully this hierarchy definition should seem familiar.

When thinking about a larger model being triggered by a smaller model and some heuristics, speculative decoding might come to mind[10](#fn-10). However if we were to have a byte-level draft model and a byte-level oracle model, it would miss out on the crucial sequence length compression and cripple the global model to have a much shorter max context length.

The core contribution of this paper is this specific separation of computation which impacts the frequency of execution and FLOPs distribution. With a full sequence T bytes and P\=4 at inference time, K\=TP where the global model is executed K times (i.e 4x less in this case) versus the local model that runs T times. The global model is also executed with T/4 patches in its context putting it in a similar position as the average subword-level token size.

In this setup, the local model is predicting each sequence element's likelihood conditioned *solely* on its respective patch (and not all previous patches!). In practice, they create K copies of the local model so that both at prefill (during inference) and training, they're able to parallelise. Its autoregressive property is upheld by, as you would also expect, the inclusion of padding and offsetting inputs to the local and global models to avoid leaking information about future positions.

One of the paper's aim is general, modality-agnostic byte modelling which is demonstrated by their evaluation against language, audio and image modelling[11](#fn-11). With respect to language modelling, they show that they outperform other byte-level models in compute-controlled experiments ([to much excitement](https://x.com/karpathy/status/1657949234535211009)) but they fail to demonstrate its performance against subword-level transformers in a compute-controlled setting. Given the paper takes on a lot, this seems to have fallen shorter on the list of priorities and so they compared in a compute-variable setting[12](#fn-12).

A follow up paper, [SpaceByte](https://arxiv.org/abs/2404.14408) demonstrates that when done in a compute-controlled setting alongside subword-level transformers, [it doesn't perform quite as competitively](https://arxiv.org/html/2404.14408v3#:~:text=0.560-,MegaByte,0.570,-SpaceByte%20) as the Megabyte authors had figured.

![](https://bear-images.sfo2.cdn.digitaloceanspaces.com/lucalp/pasted-image-20250613080243.webp)

*unit is bits-per-byte*

Given that [SpaceByte](https://arxiv.org/abs/2404.14408) is more fixated on language modelling, they invest more into conducting this benchmarking to have baselines against which to compare. Stemming from the observation that patch boundaries will occur regardless of structure (i.e in the middle of words), they introduce a modality-specific patching rule (ala [hourglass-based dynamic token pooling](https://arxiv.org/abs/2211.09761)) that gives rise to dynamic patches (i.e patch on word boundaries via *whitespace-like* byte characters). In order to handle the dynamic patches, they introduce another local model before the global transformer. In this way, they end up approximating a simpler version of the dynamic patch hourglass transformer in the "whitespace as the dynamic patch boundary predictor" configuration.

### back up for a breather

So where does this leave us?

As mentioned earlier, you'll see that all the architectures are primarily concerned with their down/upsampling method to create a compressed representation upon which they then disproportionately spend FLOPs from their budget. When designing the downsampling scheme, they're also making a choice as to whether the compressed representation will capture a fixed or dynamic width of bytes and how they're going to prevent future position information leakage.

Up until now, byte-level models seem to have found their fit in the wild for specific tasks where the granularity excels like [toxicity detection](https://x.com/YiTayML/status/1469023675135369216) which are known to outperform tokenized models on [academic benchmarks](https://arxiv.org/abs/2106.12672). For the "default" case, they haven't seen much adoption. A recent contender has built on all these previous works, invested in proper studies and achieved some interesting results that seem to be best aligned with our previously stated goal.

* * *

## Byte Latent Transformer

![](https://bear-images.sfo2.cdn.digitaloceanspaces.com/lucalp/9y3fgm.webp)

Using the set of approaches we've accumulated thus far let's break down the BLT. Much like [SpaceByte](https://arxiv.org/abs/2404.14408), it's solely focused on language modelling. Starting with a broad overview of the components, it has:

1.  Patcher 𝒫 that decides the dynamic patch boundaries for a stream of bytes
2.  Local Encoder ℰ responsible for going from bytes to patches
3.  Global Transformer 𝒢 that contextualises the patches
4.  Local Decoder 𝒟 uses byte-level info from ℰ and patches from 𝒢 to predict the next-byte of (what will be) the next patch.

Putting that all together into an animation:

For most intents and purposes, the animation above should suffice in explaining the architecture to the point that we can investigate the paper's results. For the curious, continue on, for the time-pressed - jump to [the results section](#results) or the [quirks section](#quirks) that helps build intuition via tinkering.

### mechanics

I'll refer to BLT as the local encoder/decoder + global transformer and the Patcher as a separate entity. The Patcher 𝒫's goal is similar to that in the Hourglass Transformer but rather than using a classifier trained to predict entropy-based patch boundaries, it uses the next-byte prediction of a small byte-level autoregressive LLM's to determine the boundaries via thresholds. Concretely, they're computed as the next byte entropies under the LM distribution pe over the byte vocabulary 𝒱:

H(xi)\=∑v∈𝒱pe(xi\=v∣x<i)logpe(xi\=v∣x<i)

It's trained separately from the BLT (but on the same pre-training data mix) with sliding window attention (nctx\=512). A patch boundary is then decided on the basis of either one of two thresholds:

GlobalConstraintH(xt)\>θgApprox.MonotonicConstraintH(xt)−H(xt−1)\>θr

The thresholds are calibrated on the basis of the *average desired patch size* on the pre-training data mix.

For notation, anything subscript i denotes byte-level positions while j refers to patch-level.

With patch boundaries defined, the BLT has the Local Encoder ℰ that embeds bytes bi into xi via a ℝ256×hℰ matrix. It has alternating layers of transformer blocks and multi-headed cross attention. It downsamples using xi and the patch boundaries (from 𝒫) to create patch representations pj (ala SpaceByte). 𝒢 does a bog-standard pass through the transformer to produce the contextualised oj. The Local Decoder 𝒟 also has the alternating layers just in reverse order (i.e starts with cross attention). It uses both enriched patch-level oj and intermediate byte-level x^i from ℰ to predict the next byte bi+1 of (what will be) the next patch. Prior to ℰ, a local block causal mask is applied such that byte positions can attend across patch boundaries but not across document boundaries.

Focusing on ℰ, x^i is downsampled via pooling that's then projected via a linear layer ℰC∈ℝhℰ×(hℰ×Uℰ) to become Qj where Uℰ is the number of cross attention heads and hℰ is the local encoder's embedding dimension. Qj is the patch-level query for the cross attention used in combination with byte-level Ki and Vi projections of xi. For the cross attention, a special mask is used where each Qj only attends to the K and V that corresponds to the bytes in its patch j.

On the other side, there's 𝒟 that upsamples by using x^i (i.e last transformer block output) as the byte-level queries and patch-level keys and values via projections of the enriched oj. Optionally, instead of using xi directly they imbue each position with CANINE-like n-byte hash embeddings via:

ei\=xi+∑n\=3,…,8Enhash (Hash(gi,n))

where xi is the position's byte embedding and pushed in space by each n-gram's embedding (i.e here we're adding 6 embeddings to xi). If you want an even more granular pass through the model, [the appendix](#appendix) has a concrete forward pass with shapes.

Aligned with our previous analysis of the design space, BLT is attempting to create a more efficient representation while still integrating byte-level information such that the sparingly-run latent transformer *can* model, on average, more bytes per patch OR reduce a patch down to a single byte per step on particularly difficult problems (i.e low resource languages, reverse spelling etc). Put succinctly:

> "the hypothesis that [larger models taking fewer steps on larger patches](https://arxiv.org/html/2412.09871v1#:~:text=the%20hypothesis%20that%20larger%20models%20taking%20fewer%20steps%20on%20larger%20patches%20might%20perform%20better%20than%20smaller%20models%20taking%20more%20steps) might perform better than smaller models taking more steps".

## results

*should we even care about understanding this architecture & its quirks more deeply?*

Using the criteria from earlier, let's review the loss results first.

As the old adage goes "if you're getting better results, [are you sure that you didn’t add more compute to your network?](https://nonint.com/2023/07/01/techniques-for-debugging-neural-networks/#:~:text=are%20you%20sure%20that%20you%20didn%E2%80%99t%20add%20more%20compute%20to%20your%20NN)". Hedging against this, they benchmark in compute-controlled settings for non-trivial model sizes (up to 8B and data up to 1T tokens/4T bytes). Importantly, they're comparing against both byte-level and subword-level models (i.e LLaMa 2, 3 and 3.1) which yielded the claim-to-fame graph:

![](https://bear-images.sfo2.cdn.digitaloceanspaces.com/lucalp/blt-scaling-non-compute-optimal.webp)

In this study with fixed inference FLOPS and trained beyond compute-optimal point (much like Llama 3.1), they're claiming that:

1.  BLT generally has a better scaling curve vs LLaMa 2 & 3
2.  Increasing the patch size for BLT gives better scaling curves

The second claim seems much weaker in the larger model case and they attribute this weakness to the decreasing share of total FLOPS used by the byte-level local models that seem to scale slower than the global model. If this turns out to be true, it might just be a matter of shifting inference-time FLOPS to the local models and find the right conjoined scaling method. With the first claim, we already satisfy the (1) and (3) criteria we set out so we can move on to (2) criteria.

Sticking with training an 8B beyond compute-optimal point for a 1T token dataset, they show that BLT performs better on most general-interest tasks:

![](https://bear-images.sfo2.cdn.digitaloceanspaces.com/lucalp/entropy.webp)

The downstream task performance ([each dataset explained here](https://arxiv.org/html/2412.09871v1#S5.SS3.p4.1:~:text=token%2Dbased%20models.-,6.1,Character%2DLevel%20Tasks,-A%20very%20early)) focusing on character-level tasks really shines:

![](https://bear-images.sfo2.cdn.digitaloceanspaces.com/lucalp/general-downstream.webp)

It shows that even a model trained on 16x more data is unable to get anywhere near the performance on simple noised data and basic character-level tasks. To paint a more colourful picture, some examples include:

![](https://bear-images.sfo2.cdn.digitaloceanspaces.com/lucalp/blt-cute.webp)

Going back to scaling trends, if we maintain the requirement of compute-optimality, they show matching scaling trends:

![](https://bear-images.sfo2.cdn.digitaloceanspaces.com/lucalp/compute-optimal.webp)

The obvious disclaimers here being that equal training FLOPS != equal wall clock training time given that the implementation of those FLOPS varies. This, in reality, reflects in more expensive trainings given that the [Hardware FLOPS Utilisation (HFU)](https://github.com/stas00/ml-engineering/blob/master/training/performance/README.md#glossary-and-concepts) won't be as high and therefore require longer usage. Unfortunately, they don't share any concrete details but, just as we'll [see in a bit](#patch-size-and-flops), 50% less inference FLOPS are on the table which could warrant a slightly more expensive training (much like the current trend to train models beyond compute-optimality for cheaper inference).

## quirks

### entropy-based dynamic patching

Even when considering only the entropy-based patching, there are quite a few interesting implications! They train this small byte-level LLM (aka the Patcher) on the same data mix as the BLT model hoping that low/high entropy regions of the Patcher should strongly correlate with that of the BLT[13](#fn-13). The implication of this is that the BLT ends up being able to dedicate less compute per byte to less surprising sub-sequences (since more bytes get included into a patch) or more compute to more surprising sub-sequences. Interestingly, this gives the architecture a bounded [anti-fragile](https://en.wikipedia.org/wiki/Antifragility) property in that its able to gain (dedicate more compute = better performance) from uncertainty (higher entropy) for OOD or near-OOD events.

Unlike the "less compute" case, the "more compute" case has a hard cap at 1 patch being 1 byte. Given that the less compute (higher compression) case enables the global transformer to be more efficient in the sequence dimension, it's able to squeeze more bytes into the *same* context length. Since the Patcher (that determines the surprise) is an LLM it also inherits the interesting properties of LLMs where in-context sequences become less surprising and are also further compressed! For the purpose of gaining intuition of this property, tinker around with this HF Space to see how compute will be distributed across your prompt and compare against tiktoken (GPT models) and Llama 3:

In the paper, the authors explicitly address this in-context patching with (what I'd regard) a hack of flushing the context window on newlines to avoid "entropy drift" where the designed patching behaviour departs from its desired efficiency/difficulty property but rather [impinges on performance](https://arxiv.org/html/2412.09871v1#S5.SS3.p4.1:~:text=For%20example%2C%20%E2%80%9C10%20times%2C%20with%20an%20rms%20deviation%20of%20about%E2%80%9D%20in%20the%20MMLU%20query%20is%20patched%20frequently%20the%20first%20time%20it%20is%20encountered%2C%20but%20is%20part%20of%20very%20large%20patches%20the%20next%20three%20times%2C%20which%2C%20although%20inference%20efficient%2C%20maybe%20undesirable%20for%20reasoning.) in downstream tasks such as MMLU (reasoning):

![](https://bear-images.sfo2.cdn.digitaloceanspaces.com/lucalp/patching.webp)

As the hack gets addressed, the variable-compression property seems to make this architecture appealing for its harmonious combination with reasoning. Reasoning models are foundational (haha) to current frontier models but they tend to quickly clog up their context window with long reasoning traces[14](#fn-14) and have to end up spending more tokens for handling issues due to tokenization. I'd be interested to see follow up work a BLT-like architecture with reasoning to see the impact.

Since the average patch size is determined by some threshold, the authors show that they're able to change [the patch size at inference time](https://arxiv.org/html/2412.09871v1#:~:text=However%2C%20with%20BLT,more%20inference%20steps) (i.e from a higher threshold to a lower one) so a model trained on larger patches can continue to work on smaller patches. This exists as another lever which can be used on a task-dependent basis. While nice to have at our disposal, we can anticipate it to be quite fragile and its lifetime to be tied to the eventual success of training the entropy patcher in an end-to-end fashion.

### patch size and FLOPS

When modulating patch size, it only affects the *global model*'s FLOPS contribution. The local model's FLOPS contribution won't change since they operate at the byte-level. Taking that into consideration, larger patch sizes at a smaller total BLT model size will cause the local models to make up a more significant share of the total FLOPS. Larger patch should be scaled in tandem with global model's size to properly distribute FLOPS. For this reason at patch size = 8, they're able to grow the total model parameters to be 1.7x its tokenized equivalent for the same inference budget:

![](https://bear-images.sfo2.cdn.digitaloceanspaces.com/lucalp/blt-crossover.webp)

If you're curious about this relationship between bytes, patch size and model compute, see how the FLOPS distribution changes as you change the parameters:

### n-gram hash embeddings

Surprisingly, the n-gram hash embeddings account for a total embedding table of size `shape[3_000_000, 256]` since *each* of the 6 n-gram hash groups has 500K embeddings. They aren't included in the parameter count nor in the FLOPS (as they assume its implemented as an efficient lookup table) which is in line with [OpenAI's scaling laws paper](https://arxiv.org/abs/2001.08361).

But you might be thinking, other LLMs have vocab sizes of 256K and the final linear layer demands non-trivial amounts of FLOPs, why isn't this an issue here? Since these embeddings are used only to nudge the byte-level embeddings by the neighbouring n-grams' embeddings but are never candidates for prediction themselves, the costly linear layer is avoided. They serve as imbuing byte-level positions with some context at no theoretical cost[15](#fn-15).

Given the emphasis on the compute-controlled study, I assume that the n-gram hash embeddings are a method of offloading FLOPs from the architecture via feature engineering. In the ablations, [one finds clues](https://nonint.com/2023/06/25/ablations-are-really-important/#:~:text=This%20is%20why%20I%20really%20enjoy%20reading%20the%20ablations%20sections%20of%20any%20research%20paper%3A%20it%20gives%20me%20a%20sense%20for%20what%20actually%20matters%2C%20and%20how%20much%20of%20a%20result%20is%20simply%20due%20to%20the%20implementation%20choices%20a%20researcher%20chose.). For this paper, it's [Table 9](https://arxiv.org/html/2412.09871v1#S5.SS3.p4.1:~:text=0.850-,False,0.844,-True) where they're ablating how 10 total local layers should be split amongst the local encoder/decoder. They show that with a sufficiently parametrised local encoder, the n-gram hash embeddings register no meaningful impact[16](#fn-16):

![](https://bear-images.sfo2.cdn.digitaloceanspaces.com/lucalp/ngram-ablation.webp)

Besides their ablations, they claim that at the 8B scale, going from 500K to 300K hashes per group changed performance by 0.001 bpb on 15K steps from which they highlight [how crucial they are to performance at larger scales](https://arxiv.org/html/2412.09871v1#:~:text=At%208B%20scale%20going%20from%20500K%20to%20300K%20hashes%20changed%20performance%20by%20%C2%A00.001%20bpb%20on%2015k%20steps.%20This%20indicates%20that%20hashes%20are%20vital%20to%20bringing%20the%20performance%20of%20BLT%20to%20match%20those%20of%20tokenizer%20based%20models%2C%20however%2C%20after%20300K%20hashes%2C%20there%20are%20diminishing%20returns.). I struggle to follow that conclusion given that their 8B configuration still has lℰ\=1. The authors also note that when they're training patch size 8 models, [they're using 3 encoder layers instead of 1](https://arxiv.org/html/2412.09871v1#S5.SS3.p4.1:~:text=For%20patch%20size%208%20models%2C%20we%20use%203%20encoder%20layers%20instead%20of%201) giving us an idea as to how quickly the feature engineered n-gram hash embeddings become insufficient.

### tokens to bytes in record time?

They run an experiment with initialising the global model from Llama 3.1's weights, train it for 220B tokens with a 10x lower learning rate for the global model than random init'd local models. Once complete, this model would have cumulatively been trained on 15.2T tokens. Presumably, they chose 220B tokens since that's ~1T bytes which was the crossover point in their [fixed inference scaling study](https://arxiv.org/html/2412.09871v1#S5.SS3.p4.1:~:text=400B-,1T,-Table%202%3A) against the Llama 3 4B model.

They [compare it](https://arxiv.org/html/2412.09871v1#S5.SS3.p4.1:~:text=BLT%20from,%28220B%20tokens%29) against a random init'd BLT (trained on 200B tokens) and the original Llama 3 (trained on 15T tokens) to find that it does *worse* than the thing it's init'd from *but* does better than if BLT was trained from scratch:

![](https://bear-images.sfo2.cdn.digitaloceanspaces.com/lucalp/blt-from-llama.webp)

In the interest of mitigating sunken cost and promoting adoption, this is cool that it "works" but given they come to the conclusion "byte-ifying loses *some* of the performance" and write it off as "further work needed to take full advantage" and so it isn't a feasible for a "quick" conversion to byte-level. However, if you're one of the big labs wanting to run an experiment to check feasibility of reducing your inference costs without lobotomising your model, this is decent news. If this architecture truly does contribute to unlocking an additional scaling dimension and significantly reduces inference FLOPS, it's probably less important.

## implications

With supply chain groaning to satisfy the blistering demand for intelligence across the economy, the total share of GPUs for research continues to be under pressure. Until clusters come online, it might mean that in practice less FLOPS at inference is *not only* a cost reduction measure but rather a means of [affording more of the FLOPS budget to research](https://youtu.be/yTu0ak4GoyM?si=M1hOL3NWMyuUz6K6&t=390)!

Even in the event that BLT training is less efficient w.r.t [HFU](https://github.com/stas00/ml-engineering/blob/master/training/performance/README.md#glossary-and-concepts), given this shift to serving being an increasing cost, it might be a tradeoff that carries positive ROI. Consider that mid/post-training has some amount of the training budget dedicated to it to handle lower resource languages, vocab extension, failure of tokenization etc. If a subset of that budget isn't needed and goes to a less efficient HFU BLT training, will it come out to a clear positive ROI?

If the multiscale style architecture proliferates, we'll also see serving change (similar to industrial adoption of spec decoding) given the varying frequency and memory footprint with which these byte-level and patch-level models run. Given their experiments were run against Llama 3 where they almost 2x the dense model parameters, we can only imagine what usage along with [MoE](https://arxiv.org/pdf/2412.19437) will do to cluster HBM requirements.

If sequence compression truly is pushed into an end-to-end model, sharing tokenizers as a static entity across models might be a thing of the past *but* it could be substituted for transfer-learning from patcher/encoder/decoder models into your specific domain. A few more "boring" impacts can be found in [the appendix](#boring-impacts).

## future

*what does the next iteration look like?*

With this externalised and separately trained Patcher, BLT still has some BPE-type fragility in that it depends on a component requires its [own separate tweaking](https://github.com/facebookresearch/blt/issues/106#issuecomment-2863733848) which might lead it to also become a fragile upstream dependency (but less so than BPE, hopefully).

What does a multi-modal BLT look like? If n-gram hash embeddings have to be replaced, learned modality-specific pre-processing into modality-specific embedding tables could be possible. More than that, I'd expect a deeper encoder but if the need for some additional structured signal remains, possibly some charformer-like GBST block generalised to multiple dimensions might be useful.

Multi modality will require some dynamic patch boundary predictor since the current Patcher LLM is limited to sliding window attention of 512 bytes. In the event where a 640x640 RGB image is over 1 million bytes, this probably breaks down quickly. On this point, I'd expect to see the reliance on flushing context for resolving "entropy drift" to be addressed with a more robust solution or its need completely negated as the Patcher LLM gets integrated or trained jointly with the BLT.

[In an interview](https://www.youtube.com/watch?v=8EIqHFFdccA), MEGABYTE's first author (also author on BLT) mentions how they were exploring (at that time) what a multi-scale transformer operating on some pre-tokenized input would look like. Given the lack of mention in the BLT paper, it might have turned out to be a dead end which didn't quite fit in the paper so I'd be less inclined to expect this in the next iteration.

Rather than having this external patch boundary detector and multi scale transformers to handle the dynamic patches, does the desire for adaptive compute leak into the tokenizers directly like in the [compute adaptive tokenizers paper](https://arxiv.org/pdf/2501.03120)?

Will the Bitter Lesson prevail? How far will [the path](https://storage.googleapis.com/deepmind-media/gemini/gemini_v2_5_report.pdf) of externalised tokenizers with [early fusion on existing architectures](https://arxiv.org/abs/2405.09818)take us? Will iterations that increase complexity in the architecture that decouple processing by [modality](https://arxiv.org/abs/2411.04996) or possibly by [task](https://arxiv.org/abs/2505.14683) find the right balance? If history is our guide and external factors like compute and data continue to trend upwards, we can expect that these will be stepping stones to [the Ultimate Architecture](https://cdn.prod.website-files.com/5fb8c7611599e6f11e4853cb/6311d521a48397621ae244dc_sutton%20pis.jpg) (it [might be big](https://arxiv.org/abs/1802.08864)).  

![](https://bear-images.sfo2.cdn.digitaloceanspaces.com/lucalp/young_frankenstein_end.webp)

Thanks to Christopher Fleetwood, Jochem Gietema, Philip Botros for their feedback on this post.  
  

* * *

## appendix

### BLT Granular Mechanics

I wanted to think through every step of this model at inference time to fully grok it and had this higher-level pass through already written down. I thought it'd be a shame to not share as there are probably others in the same situation since [other explanations](https://www.youtube.com/watch?v=loaTGpqfctI) aren't quite accurate. A useful mental model is that for each patch, we're squeezing out as many bytes as possible until the small autoregressive LLM's (Patcher's) entropy threshold decides that patch has been exhausted. Each patch is *not* predicted via next token prediction in lG but rather created by pooling bytes once the Patcher determines a patch boundary.

We can split decoding into two stages - inter-patch prediction and intra-patch prediction. I'm omitting the hash n-gram embeddings from the description. I'll stick to the notation from the paper, worth calling out that anything subscript by i is referencing byte-level, j is referencing patch-level.

For clarity's sake, I'll take a bs=1 scenario mid decoding where we start with the intra-patch decoding case. We have a list of patch lengths `[1, 4, 4]` determined by the small autoregressive LLM (i.e Patcher). Assuming hℰ\=256 and hG\=1024, the bytes-level local representation has `shape[1, 9, 256]` (denoted as xi) and patch-level global representation oj has already been passed through lG and has `shape[1, 2, 1024]`. Notice how we only have 2 patches even though we have 3 patch lengths.

After sampling to get `shape[1, 10]` (denoted as bi), the Patcher's entropy threshold isn't reached and so no new patch is determined. We now expect to re-use the patch-level global representation pj with `shape[1, 2, 1024]`. The next step of the decoder requires the byte-level representation bi with `shape[1, 10]` to be embedded to xi with `shape[1, 10, 256]` , go through lℰ 's transformer blocks to get the latent representation such that it can be used as the Query for the lD's Cross Attention (CA). Assuming a KV cache exists attached to the CA, we're able to do another full step through the lD's CA and transformer layers, and sample the next position so we have a new bi with`shape[1, 11]`.

If this turns out to be an end-of-patch event, we have new patch lengths `[1, 4, 6]`. This is the inter-patch prediction which requires us to get the new oj patch representations such that we have a new patch to squeeze out the next set of bytes!

It starts with the bi `shape[1, 11]` becoming xi with `shaped[1, 11, 256]`, to go through first transformer layer in lℰ into a downsample (i.e pooling), projection to become the patch-based Query for the lℰ CA followed by the rest of lℰ. This gives us a new patch-level representation pj with `shape[1, 3, 1024]` that goes through lG, gets enriched and becomes oj with `shape[1, 3, 1024]` and then a normal lD forward pass (i.e with lℰ's byte-based transformer output skip connection as the Query, oj as the patch-based KV for the lD's CA) from which we can then begin the intra patch decoding via sampling the new first byte position of this provisional patch!

### boring impacts

If tokenizers aren't standardised, what does pricing look like? It's structured exactly the same! Patches are just tokens but with variable byte compression and the concept of tokens have such strong customer understanding that if you can avoid introducing new terminology that adds complication to your customer, you would (applies to all but OpenAI with their great model naming conventions). Similarly, even though OpenAI are yet to roll out SLAs, Microsoft's Azure OpenAI Service does have [SLAs for latency](https://learn.microsoft.com/en-us/azure/ai-services/openai/faq#what-are-the-slas--service-level-agreements--in-azure-openai-) which wouldn't be materially effected.

It does, however, add a level of obscurity for all model customers given it will be less obvious how you're being charged. Given that pricing obscurity is the norm in that for reasoning, reason traces aren't shown but customers are charged for them, shows that a bit of obscurity isn't a deterrent for adoption.

### cite this post

@online{lucaperic2025,
  author  = {Luca Perić},
  title   = {The Bitter Lesson is coming for Tokenization},
  date    = {2025-06-24},
  url     = {https://lucalp.dev/bitter-lesson-tokenization-and-blt},
  urldate = {2025-07-07}
}

### footnotes

1.  It's worth noting that there is also mixed findings on these traits for different tasks as [this survey for machine translation](https://arxiv.org/abs/2110.08191) "show neither better domain robustness, nor better morphological generalization, despite being often so motivated" but do show "The only clear advantage of character models is high robustness towards source-side noise."[↩](#fnref-1)
    
2.  With an imbalanced encoder/decoder setup, inference times are going to vary on the basis of the task type where short inputs and/or long targets are more favourable for a ByT5 encoder-heavy architecture.[↩](#fnref-2)
    
3.  In this vein, [MrT5](https://arxiv.org/abs/2410.20771v3) addresses the compute problem via introducing token deletion gates to force the model to learn merging input tokens into a more compact sequence, [reducing sequence length by 50% and reaping the speedups](https://arxiv.org/html/2410.20771v3#:~:text=MrT5%20models%20that%20reduce%20the%20sequence%20length%20by%2050%25%20or%20more%20can%20achieve%2030%25%20speedup%20or%20greater). While [EvaByte](https://hkunlp.github.io/blog/2025/evabyte/) stays architecturally "simpler" by incorporating multi-byte prediction & a linearized attention variant.[↩](#fnref-3)
    
4.  for autoregressive models, guaranteeing no information leakage from future positions[↩](#fnref-4)
    
5.  Another model is also evaluated "CANINE-S" where the model uses a tokenizer for pre-training targets for the masked language modelling task.[↩](#fnref-5)
    
6.  A note in honour of the authors, this paper seems undeservingly un-cited in the MEGABYTE & Byte Latent Transformer papers[↩](#fnref-6)
    
7.  At this point, scores across blocks are independent and so they introduce an optional dot product across all scores to enable the model to consider scores across blocks when weighting the subword block representations. Meaning: P^\=softmax(PP⊤)P[↩](#fnref-7)
    
8.  It strongly resembles the work of the [Funnel Transformer](https://arxiv.org/abs/2006.03236) but doesn't target the language modelling case - the hourglass paper gets the mention instead given it works autoregressively and the follow up work that closely aligns with this post's design goal.[↩](#fnref-8)
    
9.  Follow up work [Toucan](https://arxiv.org/abs/2311.08620) removes this requirement for one of the boundary predictor cases by "reusing" the shared representations used to decode each character and they're able to speed up decoding by 2x.  
    ![](https://bear-images.sfo2.cdn.digitaloceanspaces.com/lucalp/pasted-image-20250607123159.webp)[↩](#fnref-9)
    
10.  A few of the methods for triggering the oracle's forward pass for verification are:
    
    *   static K (i.e after 4 tokens, call oracle model)
    *   tuned probability threshold (i.e generate until draft model is not confident)
    *   [staged speculative decoding](https://arxiv.org/abs/2308.04623) or methods where you remove the draft model entirely:
    
    *   [lookahead decoding](https://lmsys.org/blog/2023-11-21-lookahead-decoding/)
    *   [Medusa](https://arxiv.org/abs/2401.10774)(optionally "lossy" if using self-defined typical sampling)
    
    [↩](#fnref-10)
    
11.  Seemingly, the [first author's feed](https://x.com/liliyu_lili) is also an indication of the fixation of freeing modelling multi modality from the tyranny of tokenizers.[↩](#fnref-11)
    
12.  "We also compare with the best previously reported numbers for sub-word models. These results may be confounded by differing amounts of compute and tuning used, but show that MEGABYTE gives results competitive with state-of-theart models trained on subwords. These results suggest that MEGABYTE may allow future large language models to be tokenization-free."[↩](#fnref-12)
    
13.  they ablate [model size and context length](https://arxiv.org/html/2412.09871v1#S7.F8:~:text=We%20find%20that%20scaling%20performance%20is%20positively%20correlated%20with%20both%20these%20dimensions%20of%20the%20entropy%20model%2C%20with%20diminishing%20returns%20when%20we%20scale%20beyond%2050m%20parameters) to see the scaling laws, seeing diminishing returns at 50M.  
    ![](https://bear-images.sfo2.cdn.digitaloceanspaces.com/lucalp/patcher-ablation.webp)[↩](#fnref-13)
    
14.  this holds true regardless of whether current models are wasting tokens but this may [be less of a concern](https://arxiv.org/pdf/2503.20783) in the future as the RL matures[↩](#fnref-14)
    
15.  if they were to use the [DeepMind Chinchilla](https://arxiv.org/abs/2203.15556) paper's method, the embeddings FLOPS would be included[↩](#fnref-15)
    
16.  It's still valid that increasing encoder layer count will reduce the need for them. It's a bit hard to reason about, their n-gram ablations are all on the basis of encoder layer being set to 1. It seems odd that they don't run ablations with a heavy encoder and a light decoder with no n-gram hash embeddings.[↩](#fnref-16)
    

   26

Powered by [Bear ʕ•ᴥ•ʔ](https://bearblog.dev)