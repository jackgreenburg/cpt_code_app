import numpy as np
import random
import string
import shap
from matplotlib.colors import LinearSegmentedColormap

def shapExplainer(xgbModel, sparseMatrix):
    """
    The explainer and shap_values will be the same for all from that model
    For that reason we only want to generate it once
    """
    explainer = shap.Explainer(xgbModel)  # output_names?
    shap_values = explainer.shap_values(sparseMatrix)
    return explainer, shap_values

def genExplanation(xgbModel, text, sparseMatrixRow, words, includeAll=False, output_index=None):
    """
    Generate and display the SHAP text plot
    Since TreeExplainers in the shap module do not naturally allow for this, a Explanation object must be created
    
    output_index : int
        Index for selecting specific shap values from multi output 
    """    
    # generate explainer to get base value
    explainer, shapValues = shapExplainer(xgbModel, sparseMatrixRow)
    baseValue = explainer.expected_value
    
    # if multi output -> select proper expected value and shap output
    if isinstance(explainer.expected_value, list):
        assert output_index is not None, "Appears to be multi output but output_index is not specified."
        baseValue = baseValue[output_index]
        shapValues = shapValues[output_index]
    
    # break up report into words without spaces
    reportBroken = text.split(" ")

    shapValuesNew = []    
    if includeAll:
        reportDisplay = []
        indices = np.where(shapValues[0] != 0)[0]
        for index in indices:
            shapValuesNew.append(shapValues[0][index])
            reportDisplay.append(words[index])
        #reportDisplay = [s + "-" for s in reportDisplay]
    else:
        # loop over words in report, fetching their shap value from the previously generated shap_values
        for word in reportBroken:
            index = np.where(words == word)[0]
            if len(index) == 1:
                shapValuesNew.append(shapValues[0][index[0]])
            else:
                assert len(index) == 0
                shapValuesNew.append(0)        
        # break up report into words, adding the spaces
        reportDisplay = [e + " " for e in (text).split(" ") if e]
    
    # convert array to np.array so that .shape works in _explanation.py from the shap package
    shapValuesNew = np.array(shapValuesNew)
    
    # create explanation object
    explanation = shap.Explanation(shapValuesNew, data=reportDisplay, base_values=baseValue)    
    return explanation

def modData(report, data, shapValues, words):
    """
    Match first values of data and shapValues to report 
    """
    matter = np.array(data)
    
    reportSplit = report.split(" ")
    
    shapValuesFinal = []
    for i, word in enumerate(reportSplit):
        # check to make sure word is in the list of words that matter
        index = np.where(np.array(matter) == word)[0]  # unsure of why this is done like this
        
        # or if it is the first instance of the word in the report
        index_in_report = reportSplit.index(word)        
        
        if len(index) == 1 and index_in_report == i:
            shapValuesFinal.append(shapValues[index[0]])
        else:
            shapValuesFinal.append(0)
    total = reportSplit.copy()
     
    # add in words that don't appear in original text, but that matter
    for i, w in enumerate(matter):
        # TODO: fix <if w not in report> ... some words are two words with the singular words still appearing
        # check to see if word already included
        if w not in reportSplit:
            shapValuesFinal.append(shapValues[i])
            total.append(w)

    return shapValuesFinal, total, 

def red_transparent_blue(sV):
    colorers = []
    for l in np.linspace(1, 0, 100):
        colorers.append((30./255, 136./255, 229./255,l))
    for l in np.linspace(0, 1, 100):
        colorers.append((255./255, 13./255, 87./255,l))
    return LinearSegmentedColormap.from_list("red_transparent_blue", colorers)(sV)

# TODO: we should support text output explanations (from models that output text not numbers), this would require the force
# the force plot and the coloring to update based on mouseovers (or clicks to make it fixed) of the output text
# changed overflow="visible" to overflow="hidden"
def text(explanation, displayWords=None, num_starting_labels=0, group_threshold=1, separator='', xmin=None, xmax=None, cmax=None, value_dict=None, text_size=16):
    """ Plots an explanation of a string of text using coloring and interactive labels.

    The output is interactive HTML and you can click on any token to toggle the display of the
    SHAP value assigned to that token.

    Parameters
    ----------
    explanation : shap.Explanation() object
        List of arrays of SHAP values. Each array has the shap values for a string(# input_tokens x output_tokens).
    
    displayWords: array
        Array of same length as explanation. If this gets passed, the first first values in explanation.values and explanation.data
        must correspond with the all of the values in displayWords. 
    
    num_starting_labels : int
        Number of tokens (sorted in decending order by corresponding SHAP values) that are uncovered in the initial view. When set to 0 all tokens
        covered. 

    group_threshold : float
        The threshold used to group tokens based on interaction affects of SHAP values.

    separator : string
        The string seperator that joins tokens grouped by interation effects and unbroken string spans.

    xmin : float
        Minimum shap value bound. 

    xmax : float
        Maximum shap value bound.

    cmax : float
        Maximum absolute shap value for sample. Used for scaling colors for input tokens. 
    
    value_dict : Dict
        Dictionary of words corresponding to their SHAP value. For the purpose of coloring in the text section.
    """
    from IPython.core.display import display, HTML

    def values_min_max(values, base_values):
        """ Used to pick our axis limits.
        """
        fx = base_values + values.sum()
        xmin = fx - values[values > 0].sum()
        xmax = fx - values[values < 0].sum()
        cmax = max(abs(values.min()), abs(values.max()))
        d = xmax - xmin
        xmin -= 0.1 * d
        xmax += 0.1 * d

        return xmin, xmax, cmax
    
    # added font size functionality
    font_size = text_size
    
    # set any unset bounds
    xmin_new, xmax_new, cmax_new = values_min_max(explanation.values, explanation.base_values)
    if xmin is None:
        xmin = xmin_new
    if xmax is None:
        xmax = xmax_new
    if cmax is None:
        cmax = cmax_new


    values, clustering = unpack_shap_explanation_contents(explanation)
    tokens, values, group_sizes = process_shap_values(explanation.data, values, group_threshold, separator, clustering)
    
    # build out HTML output one word one at a time
    top_inds = np.argsort(-np.abs(values))[:num_starting_labels]
    maxv = values.max()
    minv = values.min()
    out = ""
    # ev_str = str(shap_values.base_values)
    # vsum_str = str(values.sum())
    # fx_str = str(shap_values.base_values + values.sum())
    
    uuid = ''.join(random.choices(string.ascii_lowercase, k=20))
    encoded_tokens = [t.replace("<", "&lt;").replace(">", "&gt;").replace(' ##', '') for t in tokens]

    # add the force plot
    out += svg_force_plot(values, explanation.base_values, explanation.base_values + values.sum(), encoded_tokens, uuid, xmin, xmax)
    
    value_dict = {}
    # big here
    if displayWords:
        # check valid
        for i, w in enumerate(displayWords):
            assert w == explanation.data[i]
            
            # build value dict
            if not value_dict.get(w):
                value_dict[w] = values[i]
            
        # add blank characters to end
        newTokens = [displayWords[i] if i < len(displayWords) else "" for i in range(len(explanation.data))]
    else:
        newTokens = explanation.data
    
    for i in range(len(newTokens)):
        # if header, return header, then continue to next loop
        if len(newTokens[i]) > 3:  # check to see if it could be a header
            # check [0] because it is the first and [-2] because a space gets added at the end
            if newTokens[i][0] == "!" and newTokens[i][-1] == "!":  # check if it is a header (must have been tagged as such)
                out += f"<div style=\"font-size: {font_size + 2}px; font-family: sans-serif\"><br><b>{newTokens[i][1:-1].replace('_', ' ')}</b><br></div>"
                # now continue so it doesn't get included
                continue
        
        # instead of fetching values[i], get value from some dictionary using newTokens[i]
        scaled_value = 0.5 + 0.5 * value_dict[newTokens[i]] / cmax  # here! dict
        
        # removed reference to shap/plots/_colors.py
        color = red_transparent_blue(scaled_value)
        color = (color[0]*255, color[1]*255, color[2]*255, color[3])
        
        # display the labels for the most important words
        label_display = "none"
        wrapper_display = "inline"
        if i in top_inds:
            label_display = "block"
            wrapper_display = "inline-block"
        
        # create the value_label string
        value_label = ""
        if group_sizes[i] == 1:
            value_label = str(values[i].round(3))  # here! dict
        else:
            value_label = str(values[i].round(3)) + " / " + str(group_sizes[i])  # here! dict
        
        # the HTML for this token
        out += f"""<div style='display: {wrapper_display}; text-align: center;'
    ><div style='display: {label_display}; color: #999; padding-top: 0px; font-size: {font_size - 2}px; font-family: sans-serif;'>{value_label}</div
        ><div id='_tp_{uuid}_ind_{i}'
            style='font-size: {font_size}px; display: inline; background: rgba{color}; border-radius: 3px; padding: 0px; font-family: sans-serif;'
            onclick="
            if (this.previousSibling.style.display == 'none') {{
                this.previousSibling.style.display = 'block';
                this.parentNode.style.display = 'inline-block';
            }} else {{
                this.previousSibling.style.display = 'none';
                this.parentNode.style.display = 'inline';
            }}"
            onmouseover="document.getElementById('_fb_{uuid}_ind_{i}').style.opacity = 1; document.getElementById('_fs_{uuid}_ind_{i}').style.opacity = 1;"
            onmouseout="document.getElementById('_fb_{uuid}_ind_{i}').style.opacity = 0; document.getElementById('_fs_{uuid}_ind_{i}').style.opacity = 0;"
        >{newTokens[i].replace("<", "&lt;").replace(">", "&gt;").replace(' ##', '')}</div></div>"""
        out += " " + "</div>"
    
    return out 

def process_shap_values(tokens, values, group_threshold, separator, clustering = None, return_meta_data  = False):

    # See if we got hierarchical input data. If we did then we need to reprocess the 
    # shap_values and tokens to get the groups we want to display
    M = len(tokens)
    if len(values) != M:
        
        # make sure we were given a partition tree
        if clustering is None:
            raise ValueError("The length of the attribution values must match the number of " + \
                             "tokens if shap_values.clustering is None! When passing hierarchical " + \
                             "attributions the clustering is also required.")
        
        # compute the groups, lower_values, and max_values
        groups = [[i] for i in range(M)]
        lower_values = np.zeros(len(values))
        lower_values[:M] = values[:M]
        max_values = np.zeros(len(values))
        max_values[:M] = np.abs(values[:M])
        for i in range(clustering.shape[0]):
            li = int(clustering[i,0])
            ri = int(clustering[i,1])
            groups.append(groups[li] + groups[ri])
            lower_values[M+i] = lower_values[li] + lower_values[ri] + values[M+i]
            max_values[i+M] = max(abs(values[M+i]) / len(groups[M+i]), max_values[li], max_values[ri])
    
        # compute the upper_values
        upper_values = np.zeros(len(values))
        def lower_credit(upper_values, clustering, i, value=0):
            if i < M:
                upper_values[i] = value
                return
            li = int(clustering[i-M,0])
            ri = int(clustering[i-M,1])
            upper_values[i] = value
            value += values[i]
#             lower_credit(upper_values, clustering, li, value * len(groups[li]) / (len(groups[li]) + len(groups[ri])))
#             lower_credit(upper_values, clustering, ri, value * len(groups[ri]) / (len(groups[li]) + len(groups[ri])))
            lower_credit(upper_values, clustering, li, value * 0.5)
            lower_credit(upper_values, clustering, ri, value * 0.5)

        lower_credit(upper_values, clustering, len(values) - 1)
        
        # the group_values comes from the dividends above them and below them
        group_values = lower_values + upper_values

        # merge all the tokens in groups dominated by interaction effects (since we don't want to hide those)
        new_tokens = []
        new_values = []
        group_sizes = []

        # meta data
        token_id_to_node_id_mapping = np.zeros((M,))
        collapsed_node_ids = []

        def merge_tokens(new_tokens, new_values, group_sizes, i):
            
            # return at the leaves
            if i < M and i >= 0:
                new_tokens.append(tokens[i])
                new_values.append(group_values[i])
                group_sizes.append(1)

                # meta data
                collapsed_node_ids.append(i)
                token_id_to_node_id_mapping[i] = i

            else:

                # compute the dividend at internal nodes
                li = int(clustering[i-M,0])
                ri = int(clustering[i-M,1])
                dv = abs(values[i]) / len(groups[i])
                
                # if the interaction level is too high then just treat this whole group as one token
                if dv > group_threshold * max(max_values[li], max_values[ri]):
                    new_tokens.append(separator.join([tokens[g] for g in groups[li]]) + separator + separator.join([tokens[g] for g in groups[ri]]))
                    new_values.append(group_values[i])
                    group_sizes.append(len(groups[i]))

                    # setting collapsed node ids and token id to current node id mapping metadata

                    collapsed_node_ids.append(i)
                    for g in groups[li]:
                        token_id_to_node_id_mapping[g] = i
                    
                    for g in groups[ri]:
                        token_id_to_node_id_mapping[g] = i
                    
                # if interaction level is not too high we recurse
                else:
                    merge_tokens(new_tokens, new_values, group_sizes, li)
                    merge_tokens(new_tokens, new_values, group_sizes, ri)
        merge_tokens(new_tokens, new_values, group_sizes, len(group_values) - 1)
        
        # replance the incoming parameters with the grouped versions
        tokens = np.array(new_tokens)
        values = np.array(new_values)
        group_sizes = np.array(group_sizes)

        # meta data
        token_id_to_node_id_mapping = np.array(token_id_to_node_id_mapping)
        collapsed_node_ids = np.array(collapsed_node_ids)

        M = len(tokens) 
    else:
        group_sizes = np.ones(M)
        token_id_to_node_id_mapping = np.arange(M)
        collapsed_node_ids = np.arange(M)

    if return_meta_data:
        return tokens, values, group_sizes, token_id_to_node_id_mapping, collapsed_node_ids
    else:
        return tokens, values, group_sizes

def svg_force_plot(values, base_values, fx, tokens, uuid, xmin, xmax):
    #####print(f"values={values}\nbase_values={base_values}\nfx={fx}\ntokens={tokens}\nuuid={uuid}\nxmin={xmin}\nxmax={xmax}")

    def xpos(xval):
        return 100 * (xval - xmin)  / (xmax - xmin)

    s = ''
    s += '<svg width="100%" height="80px">'
    
    ### x-axis marks ###

    # draw x axis line
    s += '<line x1="0" y1="33" x2="100%" y2="33" style="stroke:rgb(150,150,150);stroke-width:1" />'

    # draw base value
    def draw_tick_mark(xval, label=None, bold=False):
        s = ""
        s += '<line x1="%f%%" y1="33" x2="%f%%" y2="37" style="stroke:rgb(150,150,150);stroke-width:1" />' % ((xpos(xval),) * 2)
        if not bold:
            s += '<text x="%f%%" y="27" font-size="12px" fill="rgb(120,120,120)" dominant-baseline="bottom" text-anchor="middle" font-family="sans-serif">%f</text>' % (xpos(xval),xval)
        else:
            s += '<text x="%f%%" y="27" font-size="13px" style="stroke:#ffffff;stroke-width:8px;" font-weight="bold" fill="rgb(255,255,255)" dominant-baseline="bottom" text-anchor="middle">%f</text>' % (xpos(xval),xval)
            s += '<text x="%f%%" y="27" font-size="13px" font-weight="bold" fill="rgb(0,0,0)" dominant-baseline="bottom" text-anchor="middle" font-family="sans-serif">%f</text>' % (xpos(xval),xval)
        if label is not None:
            s += '<text x="%f%%" y="10" font-size="12px" fill="rgb(120,120,120)" dominant-baseline="bottom" text-anchor="middle" font-family="sans-serif">%s</text>' % (xpos(xval), label)
        return s

    s += draw_tick_mark(base_values, label="base value")
    tick_interval = (xmax - xmin) / 7
    side_buffer = (xmax - xmin) / 14
    for i in range(1,10):
        pos = base_values - i * tick_interval
        if pos < xmin + side_buffer:
            break
        s += draw_tick_mark(pos)
    for i in range(1,10):
        pos = base_values + i * tick_interval
        if pos > xmax - side_buffer:
            break
        s += draw_tick_mark(pos)
    s += draw_tick_mark(fx, bold=True, label="f(x)")
    
    
    ### Positive value marks ###
    # removed reference to shap/plots/_colors.py
    red = (255.0, 0.0, 81.08083606031792)
    light_red = (255, 195, 213)
    
    # draw base red bar
    x = fx - values[values > 0].sum()
    w = 100 * values[values > 0].sum() / (xmax - xmin)
    s += f'<rect x="{xpos(x)}%" width="{w}%" y="40" height="18" style="fill:rgb{red}; stroke-width:0; stroke:rgb(0,0,0)" />'

    # draw underline marks and the text labels
    pos = fx
    last_pos = pos
    
    # here!
    inds = [i for i in np.argsort(-np.abs(values)) if values[i] > 0]
    for i,ind in enumerate(inds):
        v = values[ind]
        pos -= v
        
        # a line under the bar to animate
        s += f'<line x1="{xpos(pos)}%" x2="{xpos(last_pos)}%" y1="60" y2="60" id="_fb_{uuid}_ind_{ind}" style="stroke:rgb{red};stroke-width:2; opacity: 0"/>'
        
        # the text label cropped and centered
        s += f'<text x="{(xpos(last_pos) + xpos(pos))/2}%" y="71" font-size="12px" id="_fs_{uuid}_ind_{ind}" fill="rgb{red}" style="opacity: 0" dominant-baseline="middle" text-anchor="middle">{values[ind].round(3)}</text>'
        
        # the text label cropped and centered
        s += f'<svg x="{xpos(pos)}%" y="40" height="20" width="{xpos(last_pos) - xpos(pos)}%">'
        s += f'  <svg x="0" y="0" width="100%" height="100%">'
        s += f'    <text x="50%" y="9" font-size="12px" fill="rgb(255,255,255)" dominant-baseline="middle" text-anchor="middle" font-family="sans-serif">{tokens[ind].strip()}</text>'
        s += f'  </svg>'
        s += f'</svg>'
        
        last_pos = pos
    
    # draw the divider padding (which covers the text near the dividers)
    pos = fx
    for i,ind in enumerate(inds):
        v = values[ind]
        pos -= v
        
        if i != 0:
            for j in range(4):
                s += f'<g transform="translate({2*j-8},0)">'
                s += f'  <svg x="{xpos(last_pos)}%" y="40" height="18" overflow="hidden" width="30">'
                s += f'    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb{red};stroke-width:2" />'
                s += f'  </svg>'
                s += f'</g>'
            
        if i + 1 != len(inds):
            for j in range(4):
                s += f'<g transform="translate({2*j-0},0)">'
                s += f'  <svg x="{xpos(pos)}%" y="40" height="18" overflow="hidden" width="30">'
                s += f'    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb{red};stroke-width:2" />'
                s += f'  </svg>'
                s += f'</g>'
        
        last_pos = pos
    
    # center padding
    s += f'<rect transform="translate(-8,0)" x="{xpos(fx)}%" y="40" width="8" height="18" style="fill:rgb{red}"/>'
        
    # cover up a notch at the end of the red bar
    pos = fx - values[values > 0].sum()
    s += f'<g transform="translate(-11.5,0)">'
    s += f'  <svg x="{xpos(pos)}%" y="40" height="18" overflow="hidden" width="30">'
    s += f'    <path d="M 10 -9 l 6 18 L 10 25 L 0 25 L 0 -9" fill="#ffffff" style="stroke:rgb(255,255,255);stroke-width:2" />'
    s += f'  </svg>'
    s += f'</g>'


    # draw the light red divider lines and a rect to handle mouseover events
    pos = fx
    last_pos = pos
    for i,ind in enumerate(inds):
        v = values[ind]
        pos -= v
        
        # divider line
        if i + 1 != len(inds):
            s += f'<g transform="translate(-1.5,0)">'
            s += f'  <svg x="{xpos(last_pos)}%" y="40" height="18" overflow="hidden" width="30">'
            s += f'    <path d="M 0 -9 l 6 18 L 0 25" fill="none" style="stroke:rgb{light_red};stroke-width:2" />'
            s += f'  </svg>'
            s += f'</g>'
        
        # mouse over rectangle
        s += f'<rect x="{xpos(pos)}%" y="40" height="20" width="{xpos(last_pos) - xpos(pos)}%"'
        s += f'      onmouseover="'
        s += f'document.getElementById(\'_tp_{uuid}_ind_{ind}\').style.textDecoration = \'underline\';'
        s += f'document.getElementById(\'_fs_{uuid}_ind_{ind}\').style.opacity = 1;'
        s += f'document.getElementById(\'_fb_{uuid}_ind_{ind}\').style.opacity = 1;'
        s += f'"'
        s += f'      onmouseout="'
        s += f'document.getElementById(\'_tp_{uuid}_ind_{ind}\').style.textDecoration = \'none\';'
        s += f'document.getElementById(\'_fs_{uuid}_ind_{ind}\').style.opacity = 0;'
        s += f'document.getElementById(\'_fb_{uuid}_ind_{ind}\').style.opacity = 0;'
        s += f'" style="fill:rgb(0,0,0,0)" />'
        
        last_pos = pos
        
    
    ### Negative value marks ###
    # removed reference to shap/plots/_colors.py
    blue = (0.0, 138.56128015770724, 250.76166088685727)
    light_blue = (208, 230, 250)
    
    # draw base blue bar
    w = 100 * -values[values < 0].sum() / (xmax - xmin)
    s += f'<rect x="{xpos(fx)}%" width="{w}%" y="40" height="18" style="fill:rgb{blue}; stroke-width:0; stroke:rgb(0,0,0)" />'

    # draw underline marks and the text labels
    pos = fx
    last_pos = pos
    inds = [i for i in np.argsort(-np.abs(values)) if values[i] < 0]
    for i,ind in enumerate(inds):
        v = values[ind]
        pos -= v
        
        # a line under the bar to animate
        s += f'<line x1="{xpos(last_pos)}%" x2="{xpos(pos)}%" y1="60" y2="60" id="_fb_{uuid}_ind_{ind}" style="stroke:rgb{blue};stroke-width:2; opacity: 0"/>'
        
        # the value text
        s += f'<text x="{(xpos(last_pos) + xpos(pos))/2}%" y="71" font-size="12px" fill="rgb{blue}" id="_fs_{uuid}_ind_{ind}" style="opacity: 0" dominant-baseline="middle" text-anchor="middle" font-family="sans-serif">{values[ind].round(3)}</text>'
        
        # the text label cropped and centered
        s += f'<svg x="{xpos(last_pos)}%" y="40" height="20" width="{xpos(pos) - xpos(last_pos)}%">'
        s += f'  <svg x="0" y="0" width="100%" height="100%">'
        s += f'    <text x="50%" y="9" font-size="12px" fill="rgb(255,255,255)" dominant-baseline="middle" text-anchor="middle" font-family="sans-serif">{tokens[ind].strip()}</text>'
        s += f'  </svg>'
        s += f'</svg>'
        
        last_pos = pos
    
    # draw the divider padding (which covers the text near the dividers)
    pos = fx
    for i,ind in enumerate(inds):
        v = values[ind]
        pos -= v
        
        if i != 0:
            for j in range(4):
                s += f'<g transform="translate({-2*j+2},0)">'
                s += f'  <svg x="{xpos(last_pos)}%" y="40" height="18" overflow="hidden" width="30">'
                s += f'    <path d="M 8 -9 l -6 18 L 8 25" fill="none" style="stroke:rgb{blue};stroke-width:2" />'
                s += f'  </svg>'
                s += f'</g>'
            
        if i + 1 != len(inds):
            for j in range(4):
                s += f'<g transform="translate(-{2*j+8},0)">'
                s += f'  <svg x="{xpos(pos)}%" y="40" height="18" overflow="hidden" width="30">'
                s += f'    <path d="M 8 -9 l -6 18 L 8 25" fill="none" style="stroke:rgb{blue};stroke-width:2" />'
                s += f'  </svg>'
                s += f'</g>'
        
        last_pos = pos
    
    # center padding
    s += f'<rect transform="translate(0,0)" x="{xpos(fx)}%" y="40" width="8" height="18" style="fill:rgb{blue}"/>'
    
    # cover up a notch at the end of the blue bar
    pos = fx - values[values < 0].sum()
    s += f'<g transform="translate(-6.0,0)">'
    s += f'  <svg x="{xpos(pos)}%" y="40" height="18" overflow="hidden" width="30">'
    s += f'    <path d="M 8 -9 l -6 18 L 8 25 L 20 25 L 20 -9" fill="#ffffff" style="stroke:rgb(255,255,255);stroke-width:2" />'
    s += f'  </svg>'
    s += f'</g>'

    # draw the light blue divider lines and a rect to handle mouseover events
    pos = fx
    last_pos = pos
    for i,ind in enumerate(inds):
        v = values[ind]
        pos -= v

        # divider line
        if i + 1 != len(inds):
            s += f'<g transform="translate(-6.0,0)">'
            s += f'  <svg x="{xpos(pos)}%" y="40" height="18" overflow="hidden" width="30">'
            s += f'    <path d="M 8 -9 l -6 18 L 8 25" fill="none" style="stroke:rgb{light_blue};stroke-width:2" />'
            s += f'  </svg>'
            s += f'</g>'
            
        # mouse over rectangle
        s += f'<rect x="{xpos(last_pos)}%" y="40" height="20" width="{xpos(pos) - xpos(last_pos)}%"'
        s += f'      onmouseover="'
        s += f'document.getElementById(\'_tp_{uuid}_ind_{ind}\').style.textDecoration = \'underline\';'
        s += f'document.getElementById(\'_fs_{uuid}_ind_{ind}\').style.opacity = 1;'
        s += f'document.getElementById(\'_fb_{uuid}_ind_{ind}\').style.opacity = 1;'
        s += f'"'
        s += f'      onmouseout="'
        s += f'document.getElementById(\'_tp_{uuid}_ind_{ind}\').style.textDecoration = \'none\';'
        s += f'document.getElementById(\'_fs_{uuid}_ind_{ind}\').style.opacity = 0;'
        s += f'document.getElementById(\'_fb_{uuid}_ind_{ind}\').style.opacity = 0;'
        s += f'" style="fill:rgb(0,0,0,0)" />'
        
        last_pos = pos

    s += '</svg>'
    
    return s

def unpack_shap_explanation_contents(shap_values):
    values = getattr(shap_values, "hierarchical_values", None)
    if values is None:
        values = shap_values.values
    clustering = getattr(shap_values, "clustering", None)

    return values, clustering

def plot(model, report, sparseMatrixRow, words, output_index=None, text_size=16):
    """This function is the only one you need to call to make the plot, everything else gets handled inside. 
    
    Parameters:
    model : model
        Machine learning model that shap can explain (only been tested with xgboost models).
    report : string
        Pathology report to be displayed under the force plot.
    sparseMatrixRow : csr_matrix
        Row of sparse matrix that the model was trained on. Must correspond with the passed report.  
    words : numpy array
        List of all words that appear in training data (?I think)
    output_index : int
        Index for selecting specific shap values from multi output 
    """
    
    expl = genExplanation(model, 
                          report,
                          sparseMatrixRow,
                          words,
                          includeAll=True,
                          output_index=output_index)
    bgShapValues, bgWords = modData(report, expl.data, expl.values, words)
    expl2 = shap.Explanation(np.array(bgShapValues), data=bgWords, base_values=expl.base_values) 

    htmlOutput = text(expl2, report.split(" "), text_size=text_size)
    
    return htmlOutput

def test(input="None"):
    print("Test 2 successful with input", input)