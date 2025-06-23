                                     Reinforcement learning, AI, and general intelligence                   

[

![Artificial Fintelligence](https://substackcdn.com/image/fetch/$s_!JwWp!,w_80,h_80,c_fill,f_auto,q_auto:good,fl_progressive:steep,g_auto/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F58c3a757-b2e7-4104-9f17-1e79c01d013c_1024x1024.png)



](/)

# [Artificial Fintelligence](/)

SubscribeSign in

#### Share this post

[

![Artificial Fintelligence](https://substackcdn.com/image/fetch/$s_!JwWp!,w_36,h_36,c_fill,f_auto,q_auto:good,fl_progressive:steep,g_auto/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F58c3a757-b2e7-4104-9f17-1e79c01d013c_1024x1024.png)

Artificial Fintelligence

Reinforcement learning and general intelligence









](#)

Copy link

Facebook

Email

Notes

More

# Reinforcement learning and general intelligence

### Epsilon random is not enough

[

![Finbarr Timbers's avatar](https://substackcdn.com/image/fetch/$s_!nAfB!,w_36,h_36,c_fill,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F1f5f239b-5582-4fb4-983a-2c42a6a3a99a_399x399.jpeg)



](https://substack.com/@finbarrtimbers)

[Finbarr Timbers](https://substack.com/@finbarrtimbers)

Jun 05, 2025

38

#### Share this post

[

![Artificial Fintelligence](https://substackcdn.com/image/fetch/$s_!JwWp!,w_36,h_36,c_fill,f_auto,q_auto:good,fl_progressive:steep,g_auto/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F58c3a757-b2e7-4104-9f17-1e79c01d013c_1024x1024.png)

Artificial Fintelligence

Reinforcement learning and general intelligence









](#)

Copy link

Facebook

Email

Notes

More

[

3

](https://www.artfintel.com/p/reinforcement-learning-and-general/comments)

3

[

Share

](javascript:void\(0\))

*A disclaimer: nothing that I say here is representing any organization other than Artificial Fintelligence. These are my views, and mine alone, although I hope that you share them after reading.*

Frontier labs are spending, in the aggregate, $100s of millions of dollars annually on data acquisition, leading to a number of startups selling data to them (Mercor, Scale, Surge, etc). The novel data, combined with reinforcement learning (RL) techniques, represents the most clear avenue to improvement, and to AGI. I am firmly convinced that scaling up RL techniques will lead to excellent products, and, eventually, AGI. A primary source of improvement over the last decade has been *scale*, as the industry has discovered one method after another that allows us to convert money into intelligence. First, bigger models. Then, more data (thereby making Alexandr Wang very rich). And now, RL.

Thanks for reading Artificial Fintelligence! Subscribe for free to receive new posts and support my work.

Subscribe

RL is the subfield of machine learning that studies algorithms which *discover new knowledge.* Reinforcement learning agents take actions in environments to systematically discover the optimal strategy (called a *policy*). An example environment is Atari: you have an environment (the Atari game) where the agent can take actions (moving in different directions, pressing the “fire” button) and the agent receives a scalar reward signal that it wants to maximize (the score). Without providing any data on how to play Atari games, RL algorithms are able to discover policies which get optimal scores in most Atari games.

The key problem in RL is the exploration/exploitation tradeoff. At each point that the agent is asked to choose an action, the agent has to decide between choosing the action which they currently think is best (”exploiting”) or trying a new action which might be better (”exploring”). This is an extremely difficult decision to get right. Consider a complicated game like Starcraft, or Dota. For any individual situation that the agent is in, how can we know what the optimal action is? It’s only after making an entire game’s worth of decisions that we are able to know if our strategy is sound. and it is only after playing many games that we are able to conclude how good we are in comparison to other players.

Large language models help significantly here, as they are much, much more sample efficient because they have incredibly strong priors. By encoding a significant fraction of human knowledge, the models are able to behave well in a variety of environments before they’ve actually received any training data.

When it comes to language modelling, most use of RL to date has been for RLHF, which is mostly used for behaviour modification. As there is (typically) no live data involved, RLHF isn’t “real” RL and does not face the exploration/exploitation tradeoff, nor does it allow for the discovery of new knowledge.

Knowledge discovery is the main unsolved problem in modern machine learning. While we've become proficient at supervised learning, we haven't yet cracked the code on how to systematically discover new knowledge, especially superhuman knowledge. For [AlphaStar](https://www.cs.cmu.edu/~sandholm/cs15-888F21/lecture16_AlphaStar%20-%20CMU%202021.pdf), for instance, they spent a lot of compute discovering new policies, as it is an extraordinarily hard problem to discover good strategies in Starcraft without prior knowledge.

Therein lies the rub; RL is simultaneously the most promising and most challenging approach we have. DeepMind invested billions of dollars in RL research with little commercial success to show for it (the [Nobel prize](https://www.nobelprize.org/prizes/chemistry/2024/press-release/), for instance, was for AlphaFold, which didn’t use RL). While RL is often the only solution for certain hard problems, it is *notoriously* difficult to implement effectively. Consider a game with discrete turns, like Chess or Go. In Go, you have on average 250 different choices at each turn, and the game lasts for 150 moves. Consequently, the game tree has approximately 250^150 nodes, or ~10^360. If [searching randomly](https://paperswithcode.com/method/epsilon-greedy-exploration) (which is how many RL algorithms explore), it is exceedingly difficult to find a reasonable trajectory in the game, which is why AlphaZero style selfplay is needed, or an AlphaGo style supervised learning phase. When we consider the LLM setting, in which typical vocabulary sizes are in the 10s to 100s of thousands of tokens, and sequence lengths can be in the 10s to 100s of thousands, the problem is made much worse. The result is a situation where RL is both necessary and yet should be considered a last resort.

Put differently, one way to think of deep learning is that it’s all about learning a good, generalizable, function approximation. In deep RL, we are approximating a value function, i.e. a function that tells us exactly how good or how bad a given state of the world would be. To improve the accuracy of the value function, we need to be able to receive data with non-trivial answers. If all we receive is the same reward (and it’s really bad), we can’t do anything. Consider a coding assistant, like Cursor’s newly released [background agent](https://docs.cursor.com/background-agent). One way to train the agent would be to give it a reward of 1 if it returns code which is merged into a pull request, and 0 otherwise. If you took a randomly initialized network, it would output gibberish, and would thus always receive a signal of 0. Once you get a model that is actually good enough to sometimes be useful to users, you can start getting meaningful signal and rapidly improve.

As an illustrative example, I have a friend who works at a large video game publisher doing RL research for games (think: EA, Sony, Microsoft, etc.). He consults with teams at the publisher’s studios that want to use RL. His first response, despite being an experienced RL practitioner with more than 2 decades of RL experience, is usually to ask if they've tried everything else, because it’s *so difficult* to get RL to work in practical settings.

The great question with reinforcement learning and language models is whether or not we’ll see results transfer to other domains, like we have seen with next token prediction. The great boon of autoregressive language models has been that it generalizes well, that is, you can train a model to predict the next token and it learns to generate text that is useful in a number of other situations. It is absolutely not clear whether that will be the case with models trained largely with RL, as RL policies tend to be overly specialized to the exact problem they were trained on. AlphaZero notoriously had problems with catastrophic forgetting; a [paper](https://arxiv.org/abs/2004.09677) that I wrote while at DeepMind showed that simple exploits existed which could consistently beat AlphaZero. This has been replicated consistently in a number of other papers. To get around this, many RL algorithms require repeatedly looking at the training data via replay buffers, which is awkward and unwieldy.

With LLMs, this is a major problem. Setting aside RL, in the open research space, we see a lot of VLMs that are trained separately from their LLMs equivalents. [DeepSeek-VL2](https://github.com/deepseek-ai/DeepSeek-VL2) is a separate family of models from [V3](https://huggingface.co/deepseek-ai/DeepSeek-V3), which is text-only, despite all the major closed source models accepting multimodal inputs. The main reason for the separation being that, in the published literature, adding multimodal capacities to LLMs sacrifices pure text performance. When we go to add in RL, we should expect the problem to become much worse, and more research to be dedicated to improving the inherent tradeoffs here.

In my experience as a practitioner, RL lives or dies based on the quality of the reward signal. One of the most able RL practitioners that I know, [Adam White](https://www.amii.ca/people/adam-white), begins all of his RL projects by first learning to predict the reward signal; and only then will try to optimize it (first predict, and then control). Systems that are optimizing complex, overfit reward models will struggle. Systems like the Allen Institute's [Tulu 3](https://arxiv.org/abs/2411.15124), which used verifiable rewards to do RL, seem like the answer, and provide motivation for the hundreds of millions of dollars that the frontier labs are spending on acquiring data.

The development of AlphaGo illustrates this paradox perfectly:

*   RL was essentially the only viable approach for achieving superhuman performance in Go
    
*   The project succeeded but required enormous resources and effort
    
*   The solution existed in a "narrow passageway" - there were likely very few variations of the AlphaGo approach that would have worked, as can be seen by the struggle that others have had replicating AlphaGo’s success in other domains.[1](https://www.artfintel.com/p/reinforcement-learning-and-general#footnote-1-160529469)
    

We're now facing a similar situation with language models:

1.  We've largely exhausted the easily accessible training data
    
2.  We need to discover new knowledge to progress further
    
3.  For superhuman knowledge in particular, we can't rely on human supervision by definition
    
4.  RL appears to be the only framework general enough to handle this challenge
    

In short, this is a call for research labs to start investing in fundamental RL research again, and in particular, on finally making progress on the exploration problem.

Subscribe

[1](https://www.artfintel.com/p/reinforcement-learning-and-general#footnote-anchor-1-160529469)

I actually can’t think of any successful applications of [MCTS](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search) to solve real world problems. Other than the AlphaGo/AlphaZero/MuZero line of work, it doesn’t seem to have led to anything, which 2017 Finbarr would have found extremely surprising.

[

![Rafi Nazamodeen's avatar](https://substackcdn.com/image/fetch/$s_!9Wfm!,w_32,h_32,c_fill,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F4ff70e26-73f2-4d06-9e22-e5f520223acc_144x144.png)



](https://substack.com/profile/180746847-rafi-nazamodeen)

[

![Richard Shannon's avatar](https://substackcdn.com/image/fetch/$s_!-NxH!,w_32,h_32,c_fill,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Fe32942b2-9690-42e0-9874-77f426aa8b9e_144x144.png)



](https://substack.com/profile/966510-richard-shannon)

[

![Nathan Lambert's avatar](https://substackcdn.com/image/fetch/$s_!RihO!,w_32,h_32,c_fill,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F8fedcdfb-e137-4f6a-9089-a46add6c6242_500x500.jpeg)



](https://substack.com/profile/10472909-nathan-lambert)

[

![Francisco Javier Arceo's avatar](https://substackcdn.com/image/fetch/$s_!y_Iq!,w_32,h_32,c_fill,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fbucketeer-e05bbc84-baa3-437e-9518-adb32be77984.s3.amazonaws.com%2Fpublic%2Fimages%2Ff7c9c751-e085-43f9-b27a-de692670c853_400x400.jpeg)



](https://substack.com/profile/100718588-francisco-javier-arceo)

[

![Jannik Schilling's avatar](https://substackcdn.com/image/fetch/$s_!giNo!,w_32,h_32,c_fill,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F21fc06d2-a1a8-4941-82a1-7d1ff32a226b_2449x2449.jpeg)



](https://substack.com/profile/99052393-jannik-schilling)

38 Likes∙

[3 Restacks](https://substack.com/note/p-160529469/restacks?utm_source=substack&utm_content=facepile-restacks)

38

#### Share this post

[

![Artificial Fintelligence](https://substackcdn.com/image/fetch/$s_!JwWp!,w_36,h_36,c_fill,f_auto,q_auto:good,fl_progressive:steep,g_auto/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F58c3a757-b2e7-4104-9f17-1e79c01d013c_1024x1024.png)

Artificial Fintelligence

Reinforcement learning and general intelligence









](#)

Copy link

Facebook

Email

Notes

More

[

3

](https://www.artfintel.com/p/reinforcement-learning-and-general/comments)

3

[

Share

](javascript:void\(0\))

#### Discussion about this post

CommentsRestacks

![User's avatar](https://substackcdn.com/image/fetch/$s_!TnFC!,w_32,h_32,c_fill,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack.com%2Fimg%2Favatars%2Fdefault-light.png)

[

![Sam Julien's avatar](https://substackcdn.com/image/fetch/$s_!DFX4!,w_32,h_32,c_fill,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack.com%2Fimg%2Favatars%2Fgreen.png)



](https://substack.com/profile/22358273-sam-julien?utm_source=comment)

[Sam Julien](https://substack.com/profile/22358273-sam-julien?utm_source=substack-feed-item)

[Jun 5](https://www.artfintel.com/p/reinforcement-learning-and-general/comment/123127971 "Jun 5, 2025, 9:25 AM")

Liked by Finbarr Timbers

Great write-up. Did you see this recent paper on RL for self-reflection? Super interesting. [https://arxiv.org/abs/2505.24726](https://arxiv.org/abs/2505.24726)

Expand full comment

[

Like (2)



](javascript:void\(0\))

Reply

Share

[1 reply by Finbarr Timbers](https://www.artfintel.com/p/reinforcement-learning-and-general/comment/123127971)

[

![Shital's avatar](https://substackcdn.com/image/fetch/$s_!mBNN!,w_32,h_32,c_fill,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F00eb8c81-fd57-4f8a-9306-a2e3a36dc39d_144x144.png)



](https://substack.com/profile/14309826-shital?utm_source=comment)

[Shital](https://substack.com/profile/14309826-shital?utm_source=substack-feed-item)

[Jun 5](https://www.artfintel.com/p/reinforcement-learning-and-general/comment/123236638 "Jun 5, 2025, 4:01 PM")

I would highly appreciate not using substack. They block ChatGPT for no reason but their profit and doesn't allow users talk to your article. They are becoming Reddit where they get free content created by user and ruin it for everyone to monetize it.

Expand full comment

[

Like



](javascript:void\(0\))

Reply

Share

[1 more comment...](https://www.artfintel.com/p/reinforcement-learning-and-general/comments)

TopLatestDiscussions

[How to hire ML engineers/researchers](https://www.artfintel.com/p/how-to-hire-ml-engineersresearchers)

[I’m going to assume that you’ve figured out how to find candidates which appear great on paper and your only problem is figuring out which of them to…](https://www.artfintel.com/p/how-to-hire-ml-engineersresearchers)

Jan 16 • 

[Finbarr Timbers](https://substack.com/@finbarrtimbers)

36

#### Share this post

[

![Artificial Fintelligence](https://substackcdn.com/image/fetch/$s_!JwWp!,w_36,h_36,c_fill,f_auto,q_auto:good,fl_progressive:steep,g_auto/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F58c3a757-b2e7-4104-9f17-1e79c01d013c_1024x1024.png)

Artificial Fintelligence

How to hire ML engineers/researchers









](#)

Copy link

Facebook

Email

Notes

More

[

4

](https://www.artfintel.com/p/how-to-hire-ml-engineersresearchers/comments)

[](javascript:void\(0\))

[Where do LLMs spend their FLOPS?](https://www.artfintel.com/p/where-do-llms-spend-their-flops)

[LLM theory, with a hint of empirical work](https://www.artfintel.com/p/where-do-llms-spend-their-flops)

Jan 29, 2024 • 

[Finbarr Timbers](https://substack.com/@finbarrtimbers)

30

#### Share this post

[

![](https://substackcdn.com/image/fetch/$s_!qsde!,w_520,h_272,c_fill,f_auto,q_auto:good,fl_progressive:steep,g_auto/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd2ff8e22-07a1-4726-b422-9e39c304ee02_1476x438.png)

![Artificial Fintelligence](https://substackcdn.com/image/fetch/$s_!JwWp!,w_36,h_36,c_fill,f_auto,q_auto:good,fl_progressive:steep,g_auto/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F58c3a757-b2e7-4104-9f17-1e79c01d013c_1024x1024.png)

Artificial Fintelligence

Where do LLMs spend their FLOPS?









](#)

Copy link

Facebook

Email

Notes

More

[

1

](https://www.artfintel.com/p/where-do-llms-spend-their-flops/comments)

[](javascript:void\(0\))

![](https://substackcdn.com/image/fetch/$s_!qsde!,w_320,h_213,c_fill,f_auto,q_auto:good,fl_progressive:steep,g_center/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fd2ff8e22-07a1-4726-b422-9e39c304ee02_1476x438.png)

[Papers I’ve read this week, Mixture of Experts edition](https://www.artfintel.com/p/papers-ive-read-this-week-mixture)

[I read a bunch of papers about conditional routing models](https://www.artfintel.com/p/papers-ive-read-this-week-mixture)

Aug 4, 2023 • 

[Finbarr Timbers](https://substack.com/@finbarrtimbers)

40

#### Share this post

[

![](https://substackcdn.com/image/fetch/$s_!33p1!,w_520,h_272,c_fill,f_auto,q_auto:good,fl_progressive:steep,g_auto/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F470d8fcb-21dc-4a37-92db-87eca77f31c1_1274x712.png)

![Artificial Fintelligence](https://substackcdn.com/image/fetch/$s_!JwWp!,w_36,h_36,c_fill,f_auto,q_auto:good,fl_progressive:steep,g_auto/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F58c3a757-b2e7-4104-9f17-1e79c01d013c_1024x1024.png)

Artificial Fintelligence

Papers I’ve read this week, Mixture of Experts edition









](#)

Copy link

Facebook

Email

Notes

More

[

3

](https://www.artfintel.com/p/papers-ive-read-this-week-mixture/comments)

[](javascript:void\(0\))

![](https://substackcdn.com/image/fetch/$s_!33p1!,w_320,h_213,c_fill,f_auto,q_auto:good,fl_progressive:steep,g_center/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F470d8fcb-21dc-4a37-92db-87eca77f31c1_1274x712.png)

See all

Ready for more?

Subscribe

© 2025 Finbarr Timbers

[Privacy](https://substack.com/privacy) ∙ [Terms](https://substack.com/tos) ∙ [Collection notice](https://substack.com/ccpa#personal-data-collected)

[Start writing](https://substack.com/signup?utm_source=substack&utm_medium=web&utm_content=footer)[Get the app](https://substack.com/app/app-store-redirect?utm_campaign=app-marketing&utm_content=web-footer-button)

[Substack](https://substack.com) is the home for great culture

#### Share

[

](#)

Copy link

Facebook

Email

Notes

More

#### Create your profile

![User's avatar](https://substackcdn.com/image/fetch/$s_!TnFC!,w_94,h_94,c_fill,f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack.com%2Fimg%2Favatars%2Fdefault-light.png)

Name\*

Email\*

Handle

Bio

Subscribe to the newsletter

I agree to Substack's [Terms of Use](https://substack.com/tos), and acknowledge its [Information Collection Notice](https://substack.com/ccpa#personal-data-collected) and [Privacy Policy](https://substack.com/privacy).

Save & Post Comment

## Only paid subscribers can comment on this post

[Already a paid subscriber? **Sign in**](https://substack.com/sign-in?redirect=%2Fp%2Freinforcement-learning-and-general&for_pub=finbarrtimbers&change_user=false)

#### Check your email

For your security, we need to re-authenticate you.

Click the link we sent to , or [click here to sign in](https://substack.com/sign-in?redirect=%2Fp%2Freinforcement-learning-and-general&for_pub=finbarrtimbers&with_password=true).