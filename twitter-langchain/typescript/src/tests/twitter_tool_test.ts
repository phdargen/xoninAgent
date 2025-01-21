import {
  TwitterAction,
  TwitterActionSchemaAny,
  TwitterAgentkit,
} from "@coinbase/cdp-agentkit-core";
import { TwitterTool } from "../twitter_tool";
import { z } from "zod";

const MOCK_DESCRIPTION = "Twitter Test Action";
const MOCK_NAME = "test_action";

describe("TwitterTool", () => {
  let mockAgentkit: jest.Mocked<TwitterAgentkit>;
  let mockAction: jest.Mocked<TwitterAction<TwitterActionSchemaAny>>;
  let twitterTool: TwitterTool<TwitterActionSchemaAny>;

  beforeEach(() => {
    mockAgentkit = {
      run: jest.fn((action, args) => action.func(mockAgentkit, args)),
    } as unknown as jest.Mocked<TwitterAgentkit>;

    mockAction = {
      name: MOCK_NAME,
      description: MOCK_DESCRIPTION,
      argsSchema: z.object({ test_param: z.string() }),
      func: jest.fn().mockResolvedValue("success"),
    } as unknown as jest.Mocked<TwitterAction<TwitterActionSchemaAny>>;

    twitterTool = new TwitterTool(mockAction, mockAgentkit);
  });

  it("should initialize with correct properties", () => {
    expect(twitterTool.name).toBe(MOCK_NAME);
    expect(twitterTool.description).toBe(MOCK_DESCRIPTION);
    expect(twitterTool.schema).toEqual(mockAction.argsSchema);
  });

  it("should execute action with valid args", async () => {
    const args = { test_param: "test" };
    const response = await twitterTool.call(args);

    expect(mockAction.func).toHaveBeenCalledWith(mockAgentkit, args);
    expect(response).toBe("success");
  });

  it("should handle schema validation errors", async () => {
    const invalidargs = { invalid_param: "test" };
    await expect(twitterTool.call(invalidargs)).rejects.toThrow();
    expect(mockAction.func).not.toHaveBeenCalled();
  });

  it("should return error message on action execution failure", async () => {
    mockAction.func.mockRejectedValue(new Error("Execution failed"));
    const args = { test_param: "test" };
    const response = await twitterTool.call(args);
    expect(response).toContain("Error executing test_action: Execution failed");
  });
});
