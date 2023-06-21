# %%
from neel.imports import *
model = HookedTransformer.from_pretrained("gelu-1l")
model("Hello World")
# %%
torch.set_grad_enabled(False)
# %%
W_pos_subset = model.W_pos[100:110][:, None, :]
W_E_norms = (model.W_E[None, :, :] + W_pos_subset).norm(dim=-1).mean(0) / np.sqrt(model.cfg.d_model)
W_E_norms[1] = (model.W_E[1] + model.W_pos[0]).norm() / np.sqrt(model.cfg.d_model)
W_E_normal = model.W_E / W_E_norms[:, None]

print(W_E_normal.shape)
print(W_E_normal.norm(dim=-1).std())

# %%
OV = model.W_V[0] @ model.W_O[0]
print(OV.shape)
# %%
# L0H7 that maps '(', ' (' to +X, and ')' to -X

OPEN_TOKEN = model.to_single_token("(")
OPEN_SPACE_TOKEN = model.to_single_token(" (")
CLOSE_TOKEN = model.to_single_token(")")
print(OPEN_TOKEN, OPEN_SPACE_TOKEN, CLOSE_TOKEN)

open_vec = W_E_normal[OPEN_TOKEN]
open_space_vec = W_E_normal[OPEN_SPACE_TOKEN]
close_vec = W_E_normal[CLOSE_TOKEN]
embed_vecs = torch.stack([open_vec, open_space_vec, close_vec])

post_head_vecs = embed_vecs @ OV[7]
print(post_head_vecs.shape)

def cos_mat_mat(mat1, mat2):
    mat1 = mat1 / mat1.norm(dim=-1, keepdim=True)
    mat2 = mat2 / mat2.norm(dim=-1, keepdim=True)
    return mat1 @ mat2.T

imshow(cos_mat_mat(embed_vecs, embed_vecs), title="Cosine Similarity of Embeddings", x=["(", " (", ")"], y=["(", " (", ")"])
imshow(cos_mat_mat(post_head_vecs, post_head_vecs), title="Cosine Similarity of Post-Head Embeddings", x=["(", " (", ")"], y=["(", " (", ")"])
imshow([cos_mat_mat(embed_vecs @ OV[i], embed_vecs @ OV[i]) for i in range(8)], title="Cosine Similarity of Post-Head Embeddings (All Heads)", x=["(", " (", ")"], y=["(", " (", ")"], facet_col=0)

# %%
bracket_dir = (post_head_vecs[0] + post_head_vecs[1])/2 - post_head_vecs[2]
bracket_dir = bracket_dir / bracket_dir.norm()

# %%
# There are a handful of neurons that strongly react to X, many of whose outputs affect the ')' logit
def cos_mat_vec(mat1, vec):
    mat1 = mat1 / mat1.norm(dim=-1, keepdim=True)
    # mat2 = mat2 / mat2.norm(dim=-1, keepdim=True)
    return mat1 @ vec
line(cos_mat_vec(model.W_in[0].T, bracket_dir), title='Cosine Sim of W_in and bracket dir')
line(model.W_in[0].T @ bracket_dir, title='Dot Product of W_in and bracket dir')
# %%
neuron_df = pd.DataFrame({"cos_in":to_numpy(cos_mat_vec(model.W_in[0].T, bracket_dir)), "dot_in":to_numpy(model.W_in[0].T @ bracket_dir), "number":np.arange(model.cfg.d_mlp)})
neuron_df.describe()

neuron_df["bias"] = to_numpy(model.b_in[0])

neuron_df["norm_in"] = to_numpy(model.W_in[0].norm(dim=0))
neuron_df["norm_out"] = to_numpy(model.W_out[0].norm(dim=-1))

neuron_df["dla"] = to_numpy(model.W_out[0] @ model.W_U[:, CLOSE_TOKEN])

px.scatter(neuron_df, x="bias", y="cos_in", hover_data=["number", "norm_in", "norm_out", "dla"], color="number", color_continuous_scale="Portland").show()
# %%
px.scatter(neuron_df, x="dla", y="cos_in", hover_data=["number", "norm_in", "norm_out", "bias"], color="number", color_continuous_scale="Portland", opacity=1.).show()

# %%
neuron_df["cos_in_abs"] = neuron_df["cos_in"].abs()
neuron_df = neuron_df.sort_values("cos_in_abs", ascending=False)
neuron_df.iloc[:10].style.background_gradient("coolwarm")

# %%
# These neurons seem to combine their non-linearities in interesting ways (?)
top_neuron_indices = neuron_df.number.iloc[:10].values

bracket_dot_in = (bracket_dir @ model.W_in[0][:, top_neuron_indices])

x = torch.linspace(-3, 3, 1000)[:, None].cuda()
line(F.gelu(x * bracket_dot_in[None, :] + model.b_in[0][top_neuron_indices]).T, x=x[:, 0], line_labels=[str(i) for i in top_neuron_indices], title="GELU of Dot Product of W_in and bracket dir")

# %%
code_data = load_dataset("NeelNanda/code-10k", split="train")
# %%
text = code_data[1]["text"]
tokens = model.to_tokens(text)[0]
print(tokens.shape)
logits, cache = model.run_with_cache(tokens)

# %%
# line([cache["scale", 0, "ln1"][0], cache["scale", 0, "ln2"][0]])

resid_mid = cache["resid_mid", 0][0]
print(resid_mid.shape)

resid_mid_normal = resid_mid / resid_mid.norm(dim=-1, keepdim=True) * np.sqrt(model.cfg.d_model)

line(resid_mid.norm(dim=-1))
line(resid_mid_normal.norm(dim=-1))

# %%

line(resid_mid_normal @ bracket_dir / np.sqrt(model.cfg.d_model))
line(resid_mid @ bracket_dir)

# %%
cumsum_brackets = [0]
for tok in model.to_str_tokens(tokens)[1:]:
    cumsum_brackets.append(cumsum_brackets[-1] + tok.count("(") - tok.count(")"))
cumsum_brackets = np.array(cumsum_brackets)
token_list = [f"{s}/{i}" for i, s in enumerate(model.to_str_tokens(tokens))]
line(cumsum_brackets, x=token_list, title="Cumulative Bracket Count")

# %%
line([cumsum_brackets, resid_mid @ bracket_dir])
scatter(cumsum_brackets, resid_mid @ bracket_dir)

# %%
px.box(x=cumsum_brackets, y=to_numpy(resid_mid @ bracket_dir))

# %%
pos_df = pd.DataFrame({"pos":np.arange(len(tokens)), "cumsum_brackets":cumsum_brackets, "resid_mid":to_numpy(resid_mid @ bracket_dir), "resid_mid_normal":to_numpy(resid_mid_normal @ bracket_dir)})
px.scatter(pos_df, x="cumsum_brackets", y="resid_mid", hover_data=["pos", "resid_mid_normal"])
# %%
log_probs = logits.log_softmax(dim=-1)
pos_df["close_log_prob"] = to_numpy(log_probs[0, :, CLOSE_TOKEN])
pos_df["open_log_prob"] = to_numpy(log_probs[0, :, OPEN_TOKEN])

line([pos_df["close_log_prob"].values, pos_df["open_log_prob"].values, pos_df["cumsum_brackets"].values], x=token_list, title="Log Prob of Close/Open Bracket")
# %%
px.box(pos_df, x="cumsum_brackets", y="close_log_prob").show()
px.box(pos_df, x="cumsum_brackets", y="open_log_prob").show()
# %%
pos_df["close_logit"] = to_numpy(logits[0, :, CLOSE_TOKEN])
pos_df["open_logit"] = to_numpy(logits[0, :, OPEN_TOKEN])

line([pos_df["close_logit"].values, pos_df["open_logit"].values, pos_df["cumsum_brackets"].values], x=token_list, title="Log Prob of Close/Open Bracket")

# %%
final_ln_scale = cache["scale", None, None][0, :, 0]
dla_labels = []
print(final_ln_scale.shape)
for ni in top_neuron_indices:
    neuron_wout_dla = model.W_out[0, ni] @ model.W_U[:, CLOSE_TOKEN]
    print(neuron_wout_dla)
    name = f"neuron_{ni}_logit"
    pos_df[name] = to_numpy(cache["post", 0][0, :, ni] * neuron_wout_dla * final_ln_scale)
    dla_labels.append(name)

px.line(pos_df, y=dla_labels)
# %%
from html import escape
import colorsys

from IPython.display import display
def create_html(strings, values):
    # escape strings to deal with tabs, newlines, etc.
    escaped_strings = [escape(s, quote=True) for s in strings]
    processed_strings = [s.replace('\n', '<br/>').replace('\t', '&emsp;').replace(" ", "&nbsp;") for s in escaped_strings]


    # scale values
    max_value = max(max(values), -min(values))
    scaled_values = [v / max_value * 0.5 for v in values]

    # create html
    html = ""
    for s, v in zip(processed_strings, scaled_values):
        if v < 0:
            hue = 0  # hue for red in HSV
        else:
            hue = 0.66  # hue for blue in HSV
        rgb_color = colorsys.hsv_to_rgb(hue, v, 1) # hsv color with hue 0.66 (blue), saturation as v, value 1
        hex_color = '#%02x%02x%02x' % (int(rgb_color[0]*255), int(rgb_color[1]*255), int(rgb_color[2]*255))
        html += f'<span style="background-color: {hex_color}; border: 1px solid lightgray; font-size: 16px; border-radius: 3px;">{s}</span>'

    display(HTML(html))
s = (create_html(["a", "b\nd", "c        d"], [1, -2, -3]))
s
# print(s)
# HTML(s)
# %%
create_html(model.to_str_tokens(tokens), 10 + pos_df["close_log_prob"].values)
# %%
for i in range(10):
    ni = top_neuron_indices[i]
    print(ni)
    create_html(model.to_str_tokens(tokens)[400:500], pos_df[f"neuron_{ni}_logit"].values[400:500])
# %%
resid_residual = resid_mid_normal - bracket_dir[None, :] * (resid_mid_normal @ bracket_dir)[:, None]
resid_bracket = bracket_dir[None, :] * (resid_mid_normal @ bracket_dir)[:, None]
neuron_residual_inputs = resid_residual @ model.W_in[0]
neuron_bracket_inputs = resid_bracket @ model.W_in[0]


line(neuron_residual_inputs[:, top_neuron_indices].T, line_labels=[str(i) for i in top_neuron_indices])
line(neuron_bracket_inputs[:, top_neuron_indices].T, line_labels=[str(i) for i in top_neuron_indices])

for i in range(5):
    scatter(x=neuron_residual_inputs[:, top_neuron_indices[i]], y=neuron_bracket_inputs[:, top_neuron_indices[i]], xaxis="Residual Input", yaxis="Bracket Input", title=f"Neuron {top_neuron_indices[i]}", color=cumsum_brackets)
# %%
ni1 = 130
ni2 = 1338
W_in = model.blocks[0].mlp.W_in
W_out = model.blocks[0].mlp.W_out
win1 = W_in[:, ni1]
win2 = W_in[:, ni2]
wout1 = W_out[ni1, :]
wout2 = W_out[ni2, :]
bin1 = model.blocks[0].mlp.b_in[ni1]
bin2 = model.blocks[0].mlp.b_in[ni2]

def cos(v, w):
    return (v @ w)/v.norm()/w.norm()
print(cos(win1, win2))
print(cos(wout1, wout2))
print(bin1, bin2)

# %%
ending_in_bracket_str = """from django.contrib import messages
from django.shortcuts import HttpResponseRedirect
from django.core.urlresolvers import reverse
from smartmin.views import SmartCRUDL, SmartCreateView, SmartReadView, SmartListView
from phoenix.apps.animals.models import Animal
from phoenix.apps.utils.upload.views import UploadView, UploadListView, UploadDeleteView
from .models import AnimalNote, AnimalDocument


class AnimalDocumentUploadView(UploadView):
    model = AnimalDocument
    delete_url = 'records.animaldocument_delete'

    def get_context_data(self, **kwargs):
        context = super(AnimalDocumentUploadView,"""

comma_tokens = model.to_tokens(ending_in_bracket_str)[0]
comma_logits, comma_cache = model.run_with_cache(comma_tokens)
comma_stacked_resid, comma_labels = comma_cache.get_full_resid_decomposition(apply_ln=True, pos_slice=[-4, -3, -2, -1], return_labels=True)
print(comma_stacked_resid.shape)
# %%
# comma_stacked_resid = comma_stacked_resid[:, 0, :, :]
comma_dla = comma_stacked_resid @ model.W_U[:, CLOSE_TOKEN]
line(comma_dla.T, line_labels=model.to_str_tokens(comma_tokens)[-4:], title="Comma DLA", x=comma_labels)

scatter(x=comma_dla[:, -1], y=comma_dla[:, -2], xaxis="DLA on ,", yaxis="DLA before ,", title="Comma DLA", hover=comma_labels, include_diag=True)
# %%


# figure out the positions to care about
dla_list = []
positions = [150, 152, 171, 184, 195, 210, 217, 224]
for pos in positions:

    # post_ln_input for MLP0 at some positions
    post_ln_input = cache["normalized", 0, "ln2"][0, pos]
    print(pos)
    # print(post_ln_input.shape)


    # vary the magnitude of the bracket direction and track close logit as a function of it (both in aggregate, and via the neurons)
    input_bracket = (post_ln_input @ bracket_dir) * bracket_dir
    input_not_bracket = post_ln_input - input_bracket
    print(input_not_bracket.norm(), input_bracket.norm(), post_ln_input.norm(), post_ln_input.norm().pow(2) - input_bracket.norm().pow(2) - input_not_bracket.norm().pow(2))

    coefs = torch.linspace(-3, 7, 100).cuda()
    varying_input = (coefs[:, None] * bracket_dir[None, :]) + input_not_bracket[None, :]

    pre = (varying_input @ W_in + model.b_in[0])
    post = F.gelu(pre)
    dla = post[:, :, None] * W_out[None, :, :] @ model.W_U[:, CLOSE_TOKEN] * cache["scale", None, None][0, pos, 0]
    print(dla.shape)

    logsumexp = logits[0, pos, :].logsumexp(dim=-1)
    dla_list.append(dla.sum(-1))
fig = line(dla_list, x=coefs, xaxis="Coefficient of Bracket Direction", yaxis="DLA on )", title="DLA on ) as a function of Bracket Direction Coefficient", return_fig=True, line_labels=[f"{token_list[i]}/{pos_df['cumsum_brackets'].iloc[i]}" for i in positions])
# fig.add_vline(x=input_bracket.norm().item(), line_width=1, line_dash="dash", line_color="black")
fig.show()

# Plot things and try to figure out what's going on

# Find a more principled way to find the positions that I care about
# %%
pos_df["bracket_coef"] = to_numpy((cache["normalized", 0, "ln2"][0, :] @ bracket_dir))
px.violin(pos_df, y="bracket_coef", x="cumsum_brackets").show()

# %%
pos = 150
post_ln_input = cache["normalized", 0, "ln2"][0, pos]
print(pos)
# print(post_ln_input.shape)


# vary the magnitude of the bracket direction and track close logit as a function of it (both in aggregate, and via the neurons)
input_bracket = (post_ln_input @ bracket_dir) * bracket_dir
input_not_bracket = post_ln_input - input_bracket
print(input_not_bracket.norm(), input_bracket.norm(), post_ln_input.norm(), post_ln_input.norm().pow(2) - input_bracket.norm().pow(2) - input_not_bracket.norm().pow(2))

coefs = torch.linspace(-3, 7, 100).cuda()
varying_input = (coefs[:, None] * bracket_dir[None, :]) + input_not_bracket[None, :]

pre = (varying_input @ W_in + model.b_in[0])
post = F.gelu(pre)
dla = post[:, :, None] * W_out[None, :, :] @ model.W_U[:, CLOSE_TOKEN] * cache["scale", None, None][0, pos, 0]

print(dla.shape)
# %%
top_neurons = neuron_df.number[:20].to_numpy()
print(top_neurons)
dla_list = []
dla_labels = []
for ni in top_neurons:
    dla_list.append(dla[:, ni])
    dla_labels.append(f"Neuron {ni}")
dla_list.append(dla[:, top_neurons].sum(-1))
dla_labels.append("Top")
dla_list.append(dla.sum(-1) - dla[:, top_neurons].sum(-1))
dla_labels.append("Residual")
dla_list.append(dla.sum(-1))
dla_labels.append("Total")

line(dla_list, line_labels=dla_labels, x=coefs, xaxis="Coefficient of Bracket Direction", yaxis="DLA on )", title="DLA on ) as a function of Bracket Direction Coefficient", return_fig=True)
# %%
code_data = load_dataset("NeelNanda/code-10k", split="train")
# %%
code_token_list = [model.to_tokens(code_data[i]["text"])[0] for i in range(1000)]
code_tokens = []
for i in range(1000):
    if len(code_token_list[i])==1024:
        code_tokens.append(code_token_list[i][:512])
    if len(code_tokens)==20:
        break
code_tokens = torch.stack(code_tokens)
print(code_tokens.shape)

# with torch.autocast():
code_logits, code_cache = model.run_with_cache(code_tokens)
# %%
cumsum_bracket_list = []
for b in range(20):
    cumsum_brackets = [0]
    for tok in model.to_str_tokens(code_tokens[b])[1:]:
        cumsum_brackets.append(cumsum_brackets[-1] + tok.count("(") - tok.count(")"))
    cumsum_brackets = np.array(cumsum_brackets)
    # token_list = [f"{s}/{i}" for i, s in enumerate(model.to_str_tokens(tokens))]
    # line(cumsum_brackets, x=token_list, title="Cumulative Bracket Count")
    cumsum_bracket_list.append(cumsum_brackets)
cumsum_bracket_list = np.stack(cumsum_bracket_list)
print(cumsum_bracket_list.shape)
line(cumsum_bracket_list)

# %%
code_close_log_prob = to_numpy(code_logits.log_softmax(dim=-1)[:, :, CLOSE_TOKEN])
code_double_close_log_prob = to_numpy(code_logits.log_softmax(dim=-1)[:, :, model.to_single_token("))")])
code_bracket_coef = to_numpy(code_cache["normalized", 0, "ln2"] @ bracket_dir)
print(code_bracket_coef.shape, code_close_log_prob.shape, code_double_close_log_prob.shape)
# %%
px.box(x=cumsum_bracket_list.flatten(), y=code_close_log_prob.flatten(), labels={"x":"Cumulative Bracket Count", "y":"Log Prob of )"}, title="Log Prob of ) as a function of Cumulative Bracket Count").show()
px.violin(x=cumsum_bracket_list.flatten(), y=code_close_log_prob.flatten(), labels={"x":"Cumulative Bracket Count", "y":"Log Prob of )"}, title="Log Prob of ) as a function of Cumulative Bracket Count").show()
px.box(x=cumsum_bracket_list.flatten(), y=code_double_close_log_prob.flatten(), labels={"x":"Cumulative Bracket Count", "y":"Log Prob of ))"}, title="Log Prob of )) as a function of Cumulative Bracket Count").show()
px.violin(x=cumsum_bracket_list.flatten(), y=code_double_close_log_prob.flatten(), labels={"x":"Cumulative Bracket Count", "y":"Log Prob of ))"}, title="Log Prob of )) as a function of Cumulative Bracket Count").show()
px.box(x=cumsum_bracket_list.flatten(), y=code_bracket_coef.flatten(), labels={"x":"Cumulative Bracket Count", "y":"Coef of bracket dir"}, title="Coef of bracket dir as a function of Cumulative Bracket Count").show()
px.violin(x=cumsum_bracket_list.flatten(), y=code_bracket_coef.flatten(), labels={"x":"Cumulative Bracket Count", "y":"Coef of bracket dir"}, title="Coef of bracket dir as a function of Cumulative Bracket Count").show()
# %%
code_X_log_prob = to_numpy(code_logits.log_softmax(dim=-1)[:, :, model.to_single_token(").")])
px.box(x=cumsum_bracket_list.flatten(), y=code_X_log_prob.flatten(), labels={"x":"Cumulative Bracket Count", "y":"Log Prob of )."}, title="Log Prob of ). as a function of Cumulative Bracket Count").show()
px.violin(x=cumsum_bracket_list.flatten(), y=code_X_log_prob.flatten(), labels={"x":"Cumulative Bracket Count", "y":"Log Prob of )."}, title="Log Prob of ). as a function of Cumulative Bracket Count").show()

# %%
mlp_out_dla = to_numpy((code_cache["mlp_out", 0] * code_cache["scale"]) @ model.W_U[:, CLOSE_TOKEN])
print(mlp_out_dla.shape)

px.box(y=mlp_out_dla.flatten(), x=cumsum_bracket_list.flatten(), labels={"x":"Cumulative Bracket Count", "y":"MLP Out DLA", title="MLP Out DLA against cumsum bracket"}).show()
px.violin(y=mlp_out_dla.flatten(), x=cumsum_bracket_list.flatten(), labels={"x":"Cumulative Bracket Count", "y":"MLP Out DLA", title="MLP Out DLA against cumsum bracket"}).show()
# %%
top_neuron_out = code_cache["post", 0][:, :, top_neurons] @ model.W_out[0, top_neurons]
print(top_neuron_out.shape)
top_neuron_dla = to_numpy(einops.einsum(top_neuron_out, model.W_U[:, CLOSE_TOKEN], code_cache["scale"][:, :, 0], "batch pos d_model, d_model, batch pos -> batch pos"))
print(top_neuron_dla.shape)

px.box(y=top_neuron_dla.flatten(), x=cumsum_bracket_list.flatten(), labels={"x":"Cumulative Bracket Count", "y":"Top Neuron DLA"}, title="Top Neuron DLA against cumsum bracket").show()
px.violin(y=top_neuron_dla.flatten(), x=cumsum_bracket_list.flatten(), labels={"x":"Cumulative Bracket Count", "y":"Top Neuron DLA"}, title="Top Neuron DLA against cumsum bracket").show()
# %%
fig1 = px.box(y=top_neuron_dla.flatten(), x=cumsum_bracket_list.flatten(), labels={"x":"Cumulative Bracket Count", "y":"Top Neuron DLA"}, title="Top Neuron DLA against cumsum bracket")
fig2 = px.box(y=mlp_out_dla.flatten(), x=cumsum_bracket_list.flatten(), labels={"x":"Cumulative Bracket Count", "y":"Top Neuron DLA"}, title="Top Neuron DLA against cumsum bracket")
big_fig = make_subplots(rows=1, cols=2, subplot_titles=["Top Neurons", "MLP Out"], shared_yaxes=True)
big_fig.add_trace(fig1.data[0], row=1, col=1)
big_fig.add_trace(fig2.data[0], row=1, col=2)
big_fig.show()

 # %%
n = 7
big_fig = make_subplots(rows=1, cols=n+1, subplot_titles=[f"N{top_neurons[i]}" for i in range(n)]+["Total"], shared_yaxes=True)
for i in range(n):
    top_single_neuron_out = code_cache["post", 0][:, :, top_neurons[i:i+1]] @ model.W_out[0, top_neurons[i:i+1]]
    print(top_single_neuron_out.shape)
    top_single_neuron_dla = to_numpy(einops.einsum(top_single_neuron_out, model.W_U[:, CLOSE_TOKEN], code_cache["scale"][:, :, 0], "batch pos d_model, d_model, batch pos -> batch pos"))
    print(top_single_neuron_dla.shape)
    fig1 = px.box(y=top_single_neuron_dla.flatten(), x=cumsum_bracket_list.flatten(), labels={"x":"Cumulative Bracket Count", "y":"Top Neuron DLA"}, title="Top Neuron DLA against cumsum bracket")
    big_fig.add_trace(fig1.data[0], row=1, col=i+1)
fig1 = px.box(y=top_neuron_dla.flatten(), x=cumsum_bracket_list.flatten(), labels={"x":"Cumulative Bracket Count", "y":"Top Neuron DLA"}, title="Top Neuron DLA against cumsum bracket")
big_fig.add_trace(fig1.data[0], row=1, col=n+1)
big_fig.show()
# %%