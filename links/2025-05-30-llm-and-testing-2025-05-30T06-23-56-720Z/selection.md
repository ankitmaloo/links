# llm and testing

- URL: [https://x.com/lateinteraction/status/1928148705145934252](https://x.com/lateinteraction/status/1928148705145934252)
- Saved on: 2025-05-30

## Selected Text

> Sigh, it's a bit of a mess.
> 
> Let me just give you guys the full nuance in one stream of consciousness since I think we'll continue to get partial interpretations that confuse everyone. All the little things I post need to always be put together in one place.
> 
> First, I have long advised from this account to simply avoid the benchmarks that the LLM trainers/providers work with. They're vastly more contaminated than you think. They're contaminated not only in terms of the task distribution, and not only in terms of train vs test, but actually also in terms of *the prompts that work*.
> 
> Second, let me elaborate on this prompts part. When a team trains an LLM to be good at math, they aren't just "making the LLM good at math". They're usually working with the same prompt template they'll use for evaluation. Because LLMs are statistical models, not logical computers, learning to be good at math in one prompt template does NOT mean that the model is good at math in "every reasonable prompt".
> 
> Not at all. It's still super sensitive. You may think that prompt sensitivity is a thing from 2022, but no, it's alive and well and has never been more severe. Good LLM post-training can alleviate this a bit, but it actually sometimes comes at the cost of model steerability. The less sensitive the model is, the harder it is to teach it nuanced tasks! It clings to one "mode" understanding.
> 
> Third, because of all this, every single result on the regular benchmarks for math/coding with recent models that are *aggressively* mid- and post-trained to be good at the exact same math/coding benchmarks are essentially useless to me. I have been saying this publicly (and privately to my students) for ages.
> 
> Why the whole community, as a whole, chooses to continue publishing essentially meaningless results BEFORE THEY STARTED THE PROJECT is beyond me. Simply do not work on areas that have this many confounders. It's not possible to do science like that, especially with a not-fully-open model. (BTW even an open model is closed. LLMs are trained, not programmed; they're too emergent.)
> 
> Fourth, what's the takeaway? The takeaway is that:
> 
> (1) RL on Qwen for math helps for spurious reasons because the model already knows this stuff, and just needs nudges to align with the downstream evaluation.
> 
> (2) But any effect of (1) above will be hugely exaggerated if there's even a slight mismatch between your (potentially EXTREMELY reasonable) prompt in your evals and the prompt used by the Qwen team. Is this your fault? IMO *no*, our community's only mistaken decision was sadly working on over-saturated meaningless math/coding benchmarks.
> 
> (Btw the meaningless-ness is always with respect to a specific model. The same benchmark could be extremely meaningful if you pick up, idk, Llama 2 or something.)
> 
> (3) Is this the fault of the Qwen team? Well, idk. It's not like it's their job to make their model convenient-for-researchers-who-want-to-study-post-training.
> 
> (4) We have a mini-paradigm crisis here. A lot of y'all say things like "turns out this RL runs was just aligning the model with the output format". Is this a bad thing or a good thing? It depends entirely on your frame. If the goal of the system is to be a programmatic component, then parsing and reliable presentation is an actual goal. If the goal of the system is user-facing and "to be good at math", then yes this is entirely a hack.
> 
> Which one is your goal? My guess is that most people haven't really thought about it.
> 
> So what's the verdict? First, Don't work on saturated stuff, as a general rule. Just don't even try; too many confounders. Second, RL has shown its value for "downstream alignment". Getting the plumbing to work and the pieces to align together for some downstream system configuration/reward.
> 
> For small models, "capability" gains so far seem to be entirely a function of mid-training, not the RL you're doing. For big models, we actually have no clear open evidence of anything just yet. All the noise you've been hearing for the last 6 months is vacuous under any scrutiny.
> 
> Sorry for the messy post (i've written all this in some 7 mins somehow?) but hope this helps.
